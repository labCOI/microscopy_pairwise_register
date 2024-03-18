import torch
import torch.nn.functional as F
import numpy as np
from EA.warp import warp_c
from time import perf_counter


# --------------------------------------------------------------------------
# 转换仿射矩阵
# --------------------------------------------------------------------------
def get_n(W, H, device="cpu"):
    """N that maps from unnormalized to normalized coordinates"""
    N = torch.zeros((3, 3), dtype=torch.float32, device=device)
    N[0, 0] = 2.0 / W
    N[0, 1] = 0
    N[1, 1] = 2.0 / H
    N[1, 0] = 0
    N[0, -1] = -1.0
    N[1, -1] = -1.0
    N[-1, -1] = 1.0
    return N


def get_n_inv(W, H, device="cpu"):
    """N that maps from normalized to unnormalized coordinates"""
    N = get_n(W, H, device)
    return torch.inverse(N)


def m2theta(M, w, h):
    """convert affine warp matrix `M` compatible with `opencv.warpAffine` to `theta` matrix
    compatible with `torch.F.affine_grid`
    """
    M_aug = torch.cat([M, torch.zeros((1, 3), device=M.device)], dim=0)
    M_aug[-1, -1] = 1.0
    N = get_n(w, h, M.device)
    N_inv = get_n_inv(w, h, M.device)
    theta = N @ M_aug @ N_inv
    theta = torch.inverse(theta)
    return theta[:2, :]


class Grid:
    def __init__(self, k=5, device="cuda"):
        self.k = k
        self.device = device

    def compute_weight(self, cluster_lst, img):
        kp_m_lst = []
        kp_label_lst = []
        for i, c in enumerate(cluster_lst):
            kp_m_lst.append(c.kp_m[c.inliers])
            kp_label_lst.append(i * torch.ones(c.inliers_num, device=self.device))
        kp_m = torch.cat(kp_m_lst)
        kp_label = torch.cat(kp_label_lst)
        height, width = img.shape[0], img.shape[1]
        x = torch.linspace(0, width - 1, width, device=self.device)
        y = torch.linspace(0, height - 1, height, device=self.device)
        y, x = torch.meshgrid([x, y])
        grid = torch.cat([x.unsqueeze(0), y.unsqueeze(0)]).permute([1, 2, 0]).reshape([-1, 1, 2])
        dist = torch.norm(grid - kp_m, dim=-1)
        dist_min, dist_min_idx = torch.topk(dist, 5, dim=1, largest=False)
        label_k = kp_label[dist_min_idx]
        weight_lst = []
        exp_dist = torch.exp(-dist_min ** 2 / width ** 2)
        for i in range(len(cluster_lst)):
            weight_i = ((label_k == i) * exp_dist).sum(dim=1)
            weight_lst.append(weight_i.unsqueeze(-1))
        weight = torch.cat(weight_lst, dim=-1).reshape(*img.shape, -1)
        weight = (weight / weight.sum(dim=-1, keepdim=True)).permute([2, 0, 1])
        return weight

    def knn_weight(self, cluster_lst, img, dist_computer=None):
        kp_m_lst = []
        # kp_label_lst = []
        height, width = img.shape[0], img.shape[1]
        for i, c in enumerate(cluster_lst):
            kp_m_lst.append(c.kp_m[c.inliers])
            # kp_label_lst.append(i * torch.ones(c.inliers_num, device=self.device))
        # kp_label = torch.cat(kp_label_lst)
        x = torch.linspace(0, width - 1, width, device=self.device)
        y = torch.linspace(0, height - 1, height, device=self.device)
        y, x = torch.meshgrid([y, x])
        grid = torch.cat([y.unsqueeze(0), x.unsqueeze(0)]).permute([1, 2, 0]).reshape([-1, 1, 2])
        weight_lst = []
        dist_min_lst = []
        dist_max_lst = []
        dist_min_idx_lst = []
        for c, kp_m in zip(cluster_lst, kp_m_lst):
            if dist_computer is not None:
                # dist = dist_computer.dist_map_fine(kp_m[:, [1, 0]], size=img.shape[:2])
                # dist = dist.reshape(-1, len(kp_m))
                dist = dist_computer[c.idx][c.inliers].permute([1, 2, 0]).reshape(-1, len(kp_m))
            else:
                dist = torch.norm(grid - kp_m[:, [1, 0]], dim=-1)  # grid和kp_m的坐标顺序正好相反(width*heigh, m)
            dist_min, dist_min_idx = torch.topk(dist, self.k, dim=1, largest=False)
            # 排除不可能的
            dist_max, _ = torch.topk(dist, self.k, dim=1)
            dist_min_lst.append(dist_min[:, -1].unsqueeze(0))  # 距离每个簇的最小距离
            dist_max_lst.append(dist_max[:, -1])  # 距离每个簇的最大距离
            dist_min_idx_lst.append(dist_min_idx.unsqueeze(0))
            # end
            p_n = (self.k / kp_m.shape[0]) / (dist_min[:, -1] ** 2 + 1e-30)
            # draw_pts(img, pts=kp_m, color=(0,255,0), thickness=10)
            # self.draw_weight((p_n/p_n.max()).reshape(1,*img.shape))
            weight_lst.append(p_n.unsqueeze(-1))
        weight = torch.cat(weight_lst, dim=-1).reshape(*img.shape, -1)  # * impossible_flag  # * weight_tr
        weight = (weight / (weight.sum(dim=-1, keepdim=True) + 1e-30)).permute([2, 0, 1])
        return weight

    def cut_trajectory(self, cluster_lst, mesh_grid, dist_min_idx_lst, trajectory):
        """
        沿着轨迹切开
        :param cluster_lst: 簇集合
        :param mesh_grid: 坐标点一一对应
        :param dist_min_idx_lst:最短的idx
        :param trajectory:轨迹
        :return:
        """
        trajectory = torch.tensor(trajectory, device=self.device)  # (600,800)
        trajectory = torch.cat(torch.where(trajectory > 128)).reshape([2, -1]).t()[::5, :]  # (m,2)
        trajectory_weight = []
        mesh_grid = mesh_grid.reshape(-1, 2)
        for c, dist_min_idx in zip(cluster_lst, dist_min_idx_lst):
            kp_nearest = c.kp_m[c.inliers][dist_min_idx][:, [1, 0]]  # (n^2, 2)
            # kp_nearest = c.centroid_m.unsqueeze(0)  # (1, 2)
            vector_x2k = kp_nearest - mesh_grid  # (n^2, 2)
            vector_x2t = trajectory.unsqueeze(1) - mesh_grid.unsqueeze(0)  # (m, n^2, 2)
            norm_vector_x2k = torch.norm(vector_x2k, dim=-1, keepdim=True)
            norm_vector_x2t = torch.norm(vector_x2t, dim=-1, keepdim=True)
            scale = norm_vector_x2t.squeeze() / norm_vector_x2k.squeeze() < 1
            vector_x2k = vector_x2k / (norm_vector_x2k + 1e-30)  # (n^2, 2)
            del norm_vector_x2k
            vector_x2t = vector_x2t / (norm_vector_x2t + 1e-30)  # (m, n^2, 2)
            del norm_vector_x2t
            angle = torch.acos(torch.clamp(torch.einsum("ij,aij->ai", vector_x2k, vector_x2t), -1.0, 1.0)) < 0.2
            trajectory_weight.append(((angle & scale).sum(dim=0) == 0).unsqueeze(-1))
        trajectory_weight = torch.cat(trajectory_weight, dim=-1)
        return trajectory_weight

    def generate_field(self, cluster_lst, img, dist_computer=None):
        height, width = img.shape[0], img.shape[1]
        # weight = self.compute_weight(cluster_lst, img)

        weight = self.knn_weight(cluster_lst, img, dist_computer)

        field_lst = []
        for c in cluster_lst:
            mat = m2theta(c.m, width, height)
            # 生成形变场
            affine_grid = F.affine_grid(mat.unsqueeze(0), (1, 1, height, width), align_corners=False)
            field_lst.append(affine_grid)

        field = (weight.unsqueeze(-1) * torch.cat(field_lst)).sum(dim=0, keepdim=True)
        field = field[0, :, :, [1, 0]].permute([2, 0, 1]).cpu().numpy().astype(np.float32)
        field[0] = (field[0] + 1) * field.shape[-2] / 2
        field[1] = (field[1] + 1) * field.shape[-1] / 2
        return field

    @staticmethod
    def draw_weight(weight):
        """
        绘制权重图
        :param weight:权重
        :return:
        """
        color_lst = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), (255, 0, 255), (0, 255, 255), (0, 0, 0)]
        color = []
        for i in range(len(weight)):
            color.append(torch.ones(1, *weight.shape[1:], 3) * torch.tensor(color_lst[i]))
        block = torch.cat(color).sum(dim=0)
        color = (weight.unsqueeze(-1).cpu() * torch.cat(color)).sum(dim=0)
        return color, block


def sample(img, field, padding_type=0):
    t1 = perf_counter()
    if padding_type == 'bound':
        img = warp_c(img, field, 'bound')
    else:
        img = warp_c(img, field)
        # img = warp(img, grid)
    # elif device == "cuda":
    #     field = field.cuda()
    #     x_field = F.interpolate(field[:, :, :, 0].unsqueeze(0), size=img.shape, mode="bilinear",
    #                             align_corners=False)
    #     y_field = F.interpolate(field[:, :, :, 1].unsqueeze(0), size=img.shape, mode="bilinear",
    #                             align_corners=False)
    #     field = torch.cat([x_field, y_field]).permute([1, 2, 3, 0]).float()
    #     img_type = img.dtype
    #     img = torch.tensor(img).unsqueeze(0).unsqueeze(0).to(field.device).float()
    #     img = F.grid_sample(img, field, align_corners=False)
    #     img = np.array(img.cpu(), dtype=img_type).squeeze()
    t2 = perf_counter()
    print("sample1:%.5f" % (t2 - t1))
    return img


# def sample2(img, field1, field2):
#     """
#     两次插值
#     首先从field2上找到field1上的坐标，然后根据这个坐标再来搞
#     :param img:
#     :param field1:
#     :param field2:
#     :return:
#     """
#     t1 = perf_counter()
#     grid1 = field1[0, :, :, [1, 0]].permute([2, 0, 1]).cpu().numpy().astype(np.float32)
#     grid2 = field2[0, :, :, [1, 0]].permute([2, 0, 1]).cpu().numpy().astype(np.float32)
#     grid1[0] = (grid1[0] + 1) * grid1.shape[-2] / 2
#     grid1[1] = (grid1[1] + 1) * grid1.shape[-1] / 2
#     grid2[0] = (grid2[0] + 1) * grid2.shape[-2] / 2
#     grid2[1] = (grid2[1] + 1) * grid2.shape[-1] / 2
#     grid_y = warp_c(grid1[0].copy(), grid2)
#     grid_x = warp_c(grid1[1].copy(), grid2)
#     grid = np.concatenate([grid_y[np.newaxis, ...], grid_x[np.newaxis, ...]])
#     img = warp_c(img, grid)
#     t2 = perf_counter()
#     print("sample2:%.5f" % (t2 - t1))
#     return img
