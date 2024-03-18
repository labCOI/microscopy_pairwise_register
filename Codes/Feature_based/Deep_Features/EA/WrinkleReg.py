from EA.ransac import *
from EA.grid import Grid, sample
from EA.dist import get_dist
import torch
from scipy.interpolate import griddata


def field_inv(field, img_mask):
    _, h, w = field.shape
    grid_x, grid_y = np.meshgrid(range(w), range(h))
    mask_flag = img_mask != 0
    field[0, mask_flag] = field[0, 0, 0]
    field[1, mask_flag] = field[1, 0, 0]
    value_y = grid_y.reshape(-1).copy()
    value_x = grid_x.reshape(-1).copy()
    value_y[mask_flag.reshape(-1)] = 0
    value_x[mask_flag.reshape(-1)] = 0
    pt = field[[1, 0], ...].reshape([2, -1]).transpose()
    field_y = griddata(pt, value_y, (grid_x, grid_y), fill_value=0)
    field_x = griddata(pt, value_x, (grid_x, grid_y), fill_value=0)
    field_i = np.concatenate([field_y[np.newaxis, ...], field_x[np.newaxis, ...]]).astype(np.float32)
    return field_i


def dist_matrix(d1, d2, is_normalized=False):
    if is_normalized:
        return 2 - 2.0 * d1 @ d2.t()
    x_norm = (d1 ** 2).sum(1).view(-1, 1)
    y_norm = (d2 ** 2).sum(1).view(1, -1)
    dist_mat = x_norm + y_norm - 2.0 * d1 @ d2.t()
    return dist_mat


class Cluster:
    def __init__(self, kp_m, kp_f, centroid_idx, cluster_idx, m=None, inliers=None):
        self.centroid_m = kp_m[centroid_idx]
        self.kp_m = kp_m[cluster_idx]
        self.kp_f = kp_f[cluster_idx]
        self.idx = cluster_idx
        self.m = m
        self.inliers = inliers
        self.inliers_num = None

    def estimate_affine(self):
        self.m, self.inliers = ransac(self.kp_m, self.kp_f, device=self.kp_m.device, rrt=3)
        self.inliers_num = self.inliers.sum()

    def estimate_affine_inv(self):
        self.m, self.inliers = ransac(self.kp_f, self.kp_m, device=self.kp_f.device, rrt=3)
        self.inliers_num = self.inliers.sum()


class WrinkleReg:
    """
    聚类、配准
    """

    def __init__(self, k=20, alpha=5, match_radius=300, K=3, img_tr=None, device="cuda", test=False):
        self.k = k
        self.match_radius = match_radius
        self.alpha = alpha
        self.device = device
        self.Grid = Grid(K, device=device)
        self.dist_computer = None
        self.kp_m = None
        self.kp_f = None
        self.test = test
        self.img_tr = img_tr
        self.kp_dist_mat = None

    def plant(self, kp_moving, kp_fixed, des1, des2, mutual=True):
        """
        匹配生长算法
        :param kp_moving:
        :param kp_fixed:
        :param des1:
        :param des2:
        :return:
        """
        with torch.no_grad():
            kp_moving = torch.tensor([kp.pt for kp in kp_moving]).to(self.device)
            kp_fixed = torch.tensor([kp.pt for kp in kp_fixed]).to(self.device)
            des1 = torch.tensor(des1).to(self.device)
            des2 = torch.tensor(des2).to(self.device)
            match1, match2, ratio = self.match_in_rad(kp_moving, kp_fixed, des1, des2,
                                                      mutual)  # 在一定的搜索框下，计算出最近邻匹配和ratio值
            kp_m = kp_moving[match1]
            kp_f = kp_fixed[match2]
            self.kp_m = kp_m
            self.kp_f = kp_f
            self.dist_computer = get_dist(self.img_tr, kp_m[:, [1, 0]].cpu().numpy(), 1, self.device)
            clusters_lst = self.init_clusters(kp_m, kp_f)
            clusters_lst = self.filter_clusters(clusters_lst)

            # clusters_lst = self.merge_clusters(clusters_lst)
            return clusters_lst

    def init_clusters(self, kp_m, kp_f):
        """
        init clusters
        cluster, and ransac in each cluster,del partial clusters.
        :param kp_m: keypoint in img_m
        :param kp_f: keypoint in img_f
        :return:
        """
        kp = torch.cat([kp_m, kp_f, self.alpha * (kp_f - kp_m)], dim=1)
        # 聚类
        kmeans = KMeans(self.k, self.dist_computer, device=self.device)
        centroids_idx, clusters_idx = kmeans.cluster(kp)  # 中心点id， 不同类点的id
        clusters_lst = []
        for cent, cluster in zip(centroids_idx, clusters_idx):
            clusters_lst.append(Cluster(kp_m, kp_f, cent, cluster))
        # 首先对每一类进行:1.删除掉部分点，2.ransac
        for cluster in clusters_lst:
            if self.test:
                cluster.estimate_affine_inv()
            else:
                cluster.estimate_affine()
        return clusters_lst
        # # ransac 聚类
        # ran_cluster = RansacCluster(self.k,rrt=2, device=self.device)
        # return ran_cluster.cluster(kp_m, kp_f)

    def generate_field(self, cluster_lst, img_m):
        # 先生成field
        # warp(img_tr)->dist_computer
        # 再生成field
        field = self.Grid.generate_field(cluster_lst, img_m, self.dist_computer)
        if self.test:
            field = field_inv(field, self.img_tr)
        elif self.img_tr is not None:
            img_tr = (sample(self.img_tr, field, 'bound') > 0) * 255
            cv2.imwrite("exp/img_tr.png", img_tr)
            del self.dist_computer
            self.dist_computer = get_dist(img_tr, self.kp_f[:, [1, 0]].cpu().numpy(), 1, self.device)
            field = self.Grid.generate_field(cluster_lst, img_m, self.dist_computer)
        return field

    def match_in_rad(self, kp1, kp2, des1, des2, mutual=True):
        des_dist_mat = dist_matrix(des1, des2)  # 先计算距离矩阵 (n1,n2)
        space_dist_mat = dist_matrix(kp1, kp2)  # 再计算空间距离 (n1,n2)
        des_dist_mat[space_dist_mat > (self.match_radius ** 2)] = 1e30  # 滤除空间距离大于一定阈值的匹配对 (n1,n2)
        dist_min12, dist_min_idx12 = torch.topk(des_dist_mat, k=2, dim=1, largest=False)  # (n1, 2) 距离最小的两个量

        dist_min21, dist_min_idx21 = torch.min(des_dist_mat, dim=0)  # (n2,) 从2到1距离最小的值
        if mutual:
            mnn = dist_min_idx21[dist_min_idx12[:, 0]] == torch.arange(kp1.shape[0], device=self.device)  # (n2,) mutual
        else:
            mnn = torch.arange(kp1.shape[0], device=self.device) > -1  # only one match
        dist_min_idx1 = torch.where(mnn)[0]  # (m)
        dist_min_idx2 = dist_min_idx12[:, 0][dist_min_idx1]
        ratio = dist_min12[:, 0][dist_min_idx1] / dist_min12[:, 1][dist_min_idx1].clamp_min_(1e-3)  # 计算ratio
        return dist_min_idx1, dist_min_idx2, ratio

    @staticmethod
    def filter_clusters(cluster_lst):
        """
        过滤类，删除掉错误的类
        :param cluster_lst:
        :return:
        """
        for i in range(len(cluster_lst))[::-1]:
            cluster = cluster_lst[i]
            if cluster.inliers_num < 10 or (cluster.inliers_num.cpu().numpy() * 100 / cluster.kp_f.shape[0]) < 2:
                del cluster_lst[i]
        return cluster_lst

    @staticmethod
    def nearest_cluster(cluster_lst, cluster):
        """
        在簇列表中找最近的簇
        :param cluster_lst:
        :param cluster:
        :return: nearest cluster idx
        """
        train = torch.cat([c.centroid_m.unsqueeze(0) for c in cluster_lst])
        query = cluster.centroid_m
        dist = torch.norm(train - query, dim=1)
        dist[dist == 0] = 1e30
        return torch.argmin(dist)

    def merge2cluster(self, cluster1, cluster2):
        kp_m = torch.cat([cluster1.kp_m, cluster2.kp_m])
        kp_f = torch.cat([cluster1.kp_f, cluster2.kp_f])
        if cluster2.inliers_num < 10:
            inliers = torch.norm(compute_affine(cluster1.m, kp_m, device=self.device) - kp_f, dim=1) < 3
            m = get_affine_mat(kp_m[inliers], kp_f[inliers], device=self.device)
            inliers = torch.norm(compute_affine(m, kp_m, device=self.device) - kp_f, dim=1) < 3
            if inliers.sum() > cluster1.inliers_num:
                return cluster1.merge(cluster2, m, inliers), True
        # 求两个点的m
        kp_m_inliers = torch.cat([cluster1.kp_m[cluster1.inliers], cluster2.kp_m[cluster2.inliers]])
        kp_f_inliers = torch.cat([cluster1.kp_f[cluster1.inliers], cluster2.kp_f[cluster2.inliers]])
        m = get_affine_mat(kp_m_inliers, kp_f_inliers, device=self.device)
        err = torch.norm(compute_affine(m, kp_m, device=self.device) - kp_f, dim=1) < 3
        if err.sum() > cluster1.inliers_num:
            m = get_affine_mat(kp_m[err], kp_f[err], device=self.device)
            inliers = torch.norm(compute_affine(m, kp_m, device=self.device) - kp_f, dim=1) < 3
            return cluster1.merge(cluster2, m, inliers), True
        else:
            return cluster1, False


class KMeans:
    def __init__(self, K=5, dist_computer=None, device="cuda"):
        """
        KMeans 算法
        :param K:
        :param dist_computer: 计算两点间距离的函数
        :param device:
        """
        self.device = device
        self.K = K
        self.dist = dist_computer
        self.dist_mat = None  # (grid_n^2, kp_num)

    def cluster(self, samples):
        """
        :param samples: (torch.tensor)输入的样本
        :return idx_updated: 质心的id_lst
                n_idx: 关键点的id_lst
        """
        n_idx = []
        # if self.dist is not None:
        #     self.dist_mat = self.dist.dist_map_cuda(samples[:, :2][:, [1, 0]])  # x,y 反
        samples = samples.to(self.device)
        num = samples.shape[0]
        if num < self.K:
            self.K = num
        dim = samples.shape[-1]
        np.random.seed(0)
        idx = torch.tensor(np.random.choice(np.arange(num), self.K, replace=False), dtype=torch.long).to(self.device)
        # 初始化
        iter_num = 0
        while True:
            idx_updated, cl = self.update(samples, idx)
            iter_num += 1
            if (idx_updated == idx).sum() == self.K or iter_num > self.K:
                for i in range(self.K):
                    n_idx.append(torch.where(cl.squeeze() == i)[0])
                return idx_updated, n_idx
            idx = idx_updated

    def update(self, samples, idx):
        """
        更新质心和聚类的集合
        :param samples: (torch.tensor)输入的样本
        :param idx:质心的id
        :return idx:质心的id (k,) range[0,n)
                cl:从属于每个质心的集合(n, 1) range[0,k)
        """
        k_center = samples[idx]
        # 分类
        dist_mat = self.dist_matrix(samples, k_center)  # (n,k)
        cl = dist_mat.argmin(dim=1, keepdim=True)  # 每个样本点从属的类 (n,1)
        cl_idx = (cl == torch.arange(self.K, device=self.device)).t()  # 给每类中的值做一个标记，横坐标是类，纵坐标是值 (k, n)
        sample = samples.unsqueeze(0).repeat([self.K, 1, 1])  # 扩充样本，并行计算 (k,n,d)
        sample = sample * cl_idx.unsqueeze(-1)  # 每类中只留下对应类的值，不属于的类置为false (k,n,d)
        k_center = sample.sum(dim=1) / (cl_idx.float().sum(dim=1, keepdim=True) + 1)  # 计算质心 (k,d)
        del sample
        # 确定质心
        dist_mat = self.dist_matrix(samples, k_center).t()  # (k,n)
        min_value, _ = dist_mat.min(dim=0)
        trash_idx = torch.where(min_value >= 99999)[0]
        cl[trash_idx] = 21
        idx = dist_mat.argmin(dim=1)  # 确定质心位置
        return idx, cl

    def dist_matrix(self, d1, d2):
        """
        :param d1:torch.float32 (n,6)
        :param d2:torch.float32 (m,6)
        :return:
        """
        if self.dist is None:
            dist_mat = dist_matrix(d1, d2)
        else:
            d2_moving = d2[:, :2]
            d1_delta = d1[:, 4:]
            d2_delta = d2[:, 4:]
            # [:,[1,0]],xy轴反转
            # dist_mat = self.dist[d2_moving[:, [1, 0]]]
            # dist_mat = dist_mat * self.dist.step_h
            dist_mat = self.dist[:, d2_moving[:, 1].type(torch.long),
                       d2_moving[:, 0].type(torch.long)]
            # + dist_matrix(d1_delta,d2_delta)
        return dist_mat


class RansacCluster:
    def __init__(self, K=20, rrt=3, device="cuda"):
        self.k = K
        self.rrt = rrt
        self.device = device

    def cluster(self, kp_m, kp_f):
        cluster_lst = []
        kp_m = kp_m.to(self.device)
        kp_f = kp_f.to(self.device)
        idx = torch.arange(len(kp_m))
        for _ in range(self.k):
            m, inliers = ransac(kp_m[idx], kp_f[idx], rrt=self.rrt, device=self.device)
            inliers_num = inliers.sum()
            if inliers_num > 10:
                cluster = Cluster(kp_m, kp_f, 0, idx, m, inliers)
                idx = idx[~inliers]
                cluster.inliers_num = inliers_num
                cluster_lst.append(cluster)
        return cluster_lst


if __name__ == "__main__":
    DEVICE = "cuda"
    import cv2
    from ImageTool.Feature import SuperPoint

    det = SuperPoint(0.015, batch_sz=16, device=DEVICE)
    det.detectAndCompute(np.random.random([128, 128]))
    # ---------------------------------------------------------
    # 读取图像
    # ---------------------------------------------------------
    img_o_fixed = cv2.imread("data/62/62.tif", cv2.IMREAD_GRAYSCALE)
    img_o_moving = cv2.imread("data/63/63.tif", cv2.IMREAD_GRAYSCALE)
    scale = 0.2
    # 轨迹
    img_tr = cv2.imread("data/63/63tr.tif", cv2.IMREAD_GRAYSCALE)
    img_tr = cv2.resize(img_tr, None, fx=scale, fy=scale)
    # end 轨迹

    # img_o_fixed = cv2.imread(PATH + "data/test/multi-part/layer159.tif", cv2.IMREAD_GRAYSCALE)
    # img_o_moving = cv2.imread(PATH + "data/test/multi-part/layer160.tif", cv2.IMREAD_GRAYSCALE)
    # scale = 0.2
    # # 轨迹
    # img_tr = cv2.imread(PATH + "data/test/multi-part/tr.tif", cv2.IMREAD_GRAYSCALE)
    # img_tr = cv2.resize(img_tr, None, fx=scale, fy=scale)
    # # end 轨迹

    # img_o_fixed = cv2.imread(PATH + "data/test/6/layer159_5_9.jpg", cv2.IMREAD_GRAYSCALE)
    # img_o_moving = cv2.imread(PATH + "data/test/6/layer160_5_9.jpg", cv2.IMREAD_GRAYSCALE)
    # scale = 0.2
    # # # 轨迹
    # img_tr = cv2.imread(PATH + "data/test/6/tr.tif", cv2.IMREAD_GRAYSCALE)
    # img_tr = cv2.resize(img_tr, None, fx=0.1, fy=0.1)
    # # end 轨迹
    # ---------------------------------------------------------
    # 预处理
    # ---------------------------------------------------------
    img_fixed = cv2.resize(img_o_fixed, None, fx=scale, fy=scale)
    img_moving = cv2.resize(img_o_moving, None, fx=scale, fy=scale)
    # ---------------------------------------------------------
    # 特征提取
    # ---------------------------------------------------------
    kp_fixed, des_fixed = det.detectAndCompute(img_fixed)
    kp_moving, des_moving = det.detectAndCompute(img_moving, mask=None)
    # ---------------------------------------------------------
    # 特征匹配
    # ---------------------------------------------------------
    reg_er = WrinkleReg(k=20, K=3, alpha=5, match_radius=500, device=DEVICE)
    c_lst = reg_er.plant(kp_moving, kp_fixed, des_moving, des_fixed)
    img_fixed = np.pad(img_fixed, [[0, (img_moving.shape[0] - img_fixed.shape[0])],
                                   [0, (img_moving.shape[1] - img_fixed.shape[1])]])

    field = reg_er.generate_field(c_lst, img_moving, img_tr, grid_n=30)

    img_rst = sample(img_o_moving, field)
