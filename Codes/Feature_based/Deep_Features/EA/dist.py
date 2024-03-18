import numpy as np
import cv2
from scipy.sparse.csgraph import floyd_warshall
import torch
import torch.nn.functional as F
from EA.wavefront import wavefront

INF = 99999


def get_dist(world_map, kp_lst, alpha=1, device="cpu"):
    if alpha == 1:
        out = wavefront(world_map, kp_lst)
    else:
        no_wall_way = wavefront(np.zeros_like(world_map), kp_lst)
        wall_way = wavefront(world_map, kp_lst)
        out = (wall_way-no_wall_way)*alpha + no_wall_way
    # out = repair_dist(out, world_map)
    out = torch.from_numpy(out).to(device)
    return out

def repair_dist(dist, world_map):
    h,w = world_map.shape
    for i in range(h):
        for j in range(w):
            if world_map[i][j] != 0:
                idy = i - 1 if i >=1 else 0
                idx = j - 1 if j >=1 else 0
                dist[:,i,j] = dist[:,idy,idx]
    return dist

# class DistPath:
#     def __init__(self, tr, grid_n=20):
#         self.grid_n = grid_n
#         self.height, self.width = tr.shape[:2]
#         self.step_h = self.height // self.grid_n
#         self.step_w = self.width // self.grid_n
#         axis = np.where(tr > 128)
#         axis = (np.clip(axis[0] // self.step_h, 0, grid_n - 1), np.clip(axis[1] // self.step_w, 0, grid_n - 1))
#         self.world_map = np.ones([self.grid_n, self.grid_n], dtype=np.int32)
#         self.world_map[axis] = INF
#         self.tr_mask = (tr > 128) * INF
#         # self.dist_matrix = self.init_matrix()
#         # self.dist_matrix = np.clip(floyd_warshall(self.dist_matrix, unweighted=False), 0, INF)
#         print("finished")
#
#     def __idx__(self, x, y):
#         return (x * self.grid_n + y).astype(np.int)
#
#     def __dist__(self, x_int, x_float):
#         """
#         整数x_int到小数x_float之间的距离
#         :param x_int:
#         :param x_float:
#         :return:
#         """
#         x_int = x_int.clip(0, self.grid_n - 1)
#         x_float = x_float.clip(0, self.grid_n - 1)
#         dist_tl = self.dist_matrix[
#             self.__idx__(x_int[0], x_int[1]), self.__idx__(np.floor(x_float[0]), np.floor(x_float[1]))]
#         dist_tr = self.dist_matrix[
#             self.__idx__(x_int[0], x_int[1]), self.__idx__(np.floor(x_float[0]), np.ceil(x_float[1]))]
#         dist_bl = self.dist_matrix[
#             self.__idx__(x_int[0], x_int[1]), self.__idx__(np.ceil(x_float[0]), np.floor(x_float[1]))]
#         dist_br = self.dist_matrix[
#             self.__idx__(x_int[0], x_int[1]), self.__idx__(np.ceil(x_float[0]), np.ceil(x_float[1]))]
#         delta_x = np.modf(x_float[1])[0]
#         delta_y = np.modf(x_float[0])[0]
#         dist_t = (1 - delta_x) * dist_tl + delta_x * dist_tr
#         dist_b = (1 - delta_x) * dist_bl + delta_x * dist_br
#         dist_x = (1 - delta_y) * dist_t + delta_y * dist_b
#         return dist_x
#
#     def get_mask(self):
#         mask = np.ones([self.grid_n, self.grid_n], dtype=np.uint8)
#         mask[self.world_map == INF] = 0
#         return mask
#
#     def get_grid_global_coordinates(self, i, axis=0):
#         if axis == 0:
#             return int(i + 0.5) * self.step_h
#         else:
#             return int(i + 0.5) * self.step_w
#
#     def init_matrix(self):
#         # idx = i*grid + j
#         dist_matrix = np.ones([self.grid_n * self.grid_n, self.grid_n * self.grid_n], dtype=np.int) * INF
#         for i in range(self.grid_n):
#             for j in range(self.grid_n):
#                 dist_matrix[i * self.grid_n + j, i * self.grid_n + j] = 0
#                 if i - 1 >= 0:
#                     dist_matrix[i * self.grid_n + j, (i - 1) * self.grid_n + j] = self.world_map[i - 1, j]
#                 if i + 1 < self.grid_n:
#                     dist_matrix[i * self.grid_n + j, (i + 1) * self.grid_n + j] = self.world_map[i + 1, j]
#                 if j - 1 >= 0:
#                     dist_matrix[i * self.grid_n + j, i * self.grid_n + j - 1] = self.world_map[i, j - 1]
#                 if j + 1 < self.grid_n:
#                     dist_matrix[i * self.grid_n + j, i * self.grid_n + j + 1] = self.world_map[i, j + 1]
#         # 斜边四个方向
#         # for i in range(self.grid_n):
#         #     for j in range(self.grid_n):
#         #         dist_matrix[i * self.grid_n + j, i * self.grid_n + j] = 0
#         #         if i - 1 >= 0 and j - 1 >= 0:
#         #             dist_matrix[i * self.grid_n + j, (i - 1) * self.grid_n + (j - 1)] = 1.414 * self.path_cost[
#         #                 i - 1, j - 1]
#         #         if i + 1 < self.grid_n and j - 1 >= 0:
#         #             dist_matrix[i * self.grid_n + j, (i + 1) * self.grid_n + (j - 1)] = 1.414 * self.path_cost[
#         #                 i + 1, j - 1]
#         #         if i - 1 >= 0 and j + 1 <= self.grid_n:
#         #             dist_matrix[i * self.grid_n + j, (i - 1) * self.grid_n + (j - 1)] = 1.414 * self.path_cost[
#         #                 i - 1, j - 1]
#         #         if i + 1 < self.grid_n and j + 1 < self.grid_n:
#         #             dist_matrix[i * self.grid_n + j, (i + 1) * self.grid_n + (j + 1)] = 1.414 * self.path_cost[
#         #                 i + 1, j + 1]
#         return dist_matrix
#
#     def dist_map(self, x, dist_matrix=None, size=None):
#         """
#         输出点集x到平面上每个点之间的距离
#         :param x:
#         :param dist_matrix: 距离矩阵
#         :param size:
#         :return:
#         """
#         if dist_matrix is None:
#             dist_matrix = self.dist_matrix
#         device = x.device
#         x = (x.float()).cpu().numpy()
#         if size is None:
#             x[:, 0] = x[:, 0] / self.step_h
#             x[:, 1] = x[:, 1] / self.step_w
#         else:
#             x[:, 0] = x[:, 0] / (size[0] // self.grid_n)
#             x[:, 1] = x[:, 1] / (size[1] // self.grid_n)
#         x = x.clip(0, self.grid_n - 1)
#         x_tl = dist_matrix[:, (np.floor(x[:, 0]) * self.grid_n + np.floor(x[:, 1])).astype(np.int)]
#         x_tr = dist_matrix[:, (np.floor(x[:, 0]) * self.grid_n + np.ceil(x[:, 1])).astype(np.int)]
#         x_bl = dist_matrix[:, (np.ceil(x[:, 0]) * self.grid_n + np.floor(x[:, 1])).astype(np.int)]
#         x_br = dist_matrix[:, (np.ceil(x[:, 0]) * self.grid_n + np.ceil(x[:, 1])).astype(np.int)]
#         delta_x = np.modf(x[:, 1])[0][np.newaxis, :]
#         delta_y = np.modf(x[:, 0])[0][np.newaxis, :]
#         x_t = (1 - delta_x) * x_tl + delta_x * x_tr
#         x_b = (1 - delta_x) * x_bl + delta_x * x_br
#         xx = (1 - delta_y) * x_t + delta_y * x_b
#         xx = xx.reshape([self.grid_n, self.grid_n, -1])
#         if size is not None:
#             # 切掉最外面一层边
#             bound_rate = 1 / self.grid_n
#             delta_sz0 = int(size[0] * bound_rate)
#             delta_sz1 = int(size[1] * bound_rate)
#             size1 = (size[0] + delta_sz0 * 2, size[1] + delta_sz1 * 2)
#             xx = torch.tensor(xx, device=device).permute([2, 0, 1]).unsqueeze(0)
#             xx = F.interpolate(xx, size1, mode='bilinear')
#             xx = xx.squeeze(0).permute([1, 2, 0])
#             xx = xx[delta_sz0:delta_sz0 + size[0], delta_sz1:delta_sz1 + size[1], :]
#         return xx
#
#     def dist_map_pixel(self, x):
#         """
#         输出点集x到平面上每个点之间的距离, 精细距离
#         :param x:tensor(n,2)
#         :return: #(pt,h,w)
#         """
#         out = wavefront(self.tr_mask, x.cpu().numpy())
#         return out
#
#     """
#     .           .    .      .           .
#     x1          x2   x      x3          x4
#                 |    |
#                 --dx--
#     inter   x = dx*x2 + (1-dx)*x3
#     extern  x = dx*(x2-x1) + x2
#             x = (1-dx)*(x3-x4) + x3
#     """
#
#     def dist_map_cuda(self, x_set, dist_matrix=None, size=None):
#         """
#         输出点集x到平面上每个点之间的距离
#         :param x_set: tensor(n,2)
#         :param dist_matrix: 距离矩阵 numpy.float(grid_n^2,grid_n^2)
#         :param size:
#         :return: xx
#         """
#         x = x_set.clone()
#         if dist_matrix is None:
#             dist_matrix = torch.tensor(self.dist_matrix).to(x.device)
#         if size is None:
#             x[:, 0] = x[:, 0] / self.step_h
#             x[:, 1] = x[:, 1] / self.step_w
#         else:
#             x[:, 0] = x[:, 0] / (size[0] // self.grid_n)
#             x[:, 1] = x[:, 1] / (size[1] // self.grid_n)
#         x = x.clip(0, self.grid_n - 1)
#
#         x_tl = dist_matrix[:, (torch.floor(x[:, 0]) * self.grid_n + torch.floor(x[:, 1])).type(torch.long)]
#         x_tr = dist_matrix[:, (torch.floor(x[:, 0]) * self.grid_n + torch.ceil(x[:, 1])).type(torch.long)]
#         x_bl = dist_matrix[:, (torch.ceil(x[:, 0]) * self.grid_n + torch.floor(x[:, 1])).type(torch.long)]
#         x_br = dist_matrix[:, (torch.ceil(x[:, 0]) * self.grid_n + torch.ceil(x[:, 1])).type(torch.long)]
#         # if xtl=inf then xtl=xtr, if xbl=inf  then xbl= xbr
#         x_tl[x_tl == INF] = x_tr[x_tl == INF]
#         x_tr[x_tr == INF] = x_tl[x_tr == INF]
#         x_bl[x_bl == INF] = x_br[x_bl == INF]
#         x_br[x_br == INF] = x_bl[x_br == INF]
#
#         delta_x = torch.fmod(x[:, 1], 1).unsqueeze(0)
#         delta_y = torch.fmod(x[:, 0], 1).unsqueeze(0)
#         x_t = (1 - delta_x) * x_tl + delta_x * x_tr
#         x_b = (1 - delta_x) * x_bl + delta_x * x_br
#         # if xt=inf then xt=xb, if xb=inf  then xb= xt
#         x_t[x_t == INF] = x_b[x_t == INF]
#         x_b[x_b == INF] = x_t[x_b == INF]
#         xx = (1 - delta_y) * x_t + delta_y * x_b
#         if size is not None:
#             # 切掉最外面一层边
#             xx = xx.reshape([self.grid_n, self.grid_n, -1])
#             bound_rate = 1 / self.grid_n
#             delta_sz0 = int(size[0] * bound_rate)
#             delta_sz1 = int(size[1] * bound_rate)
#             size1 = (size[0] + delta_sz0 * 2, size[1] + delta_sz1 * 2)
#             xx = xx.permute([2, 0, 1]).unsqueeze(0)
#             xx = F.interpolate(xx, size1, mode='bilinear')
#             xx = xx.squeeze(0).permute([1, 2, 0])
#             xx = xx[delta_sz0:delta_sz0 + size[0], delta_sz1:delta_sz1 + size[1], :]
#         return xx
#
#     def dist_points(self, x1, x2):
#         """
#         输出点x1,x2之间的距离
#         :param x1: numpy
#         :param x2: numpy
#         :return:
#         """
#         x1[0] = x1[0] / self.step_h
#         x1[1] = x1[1] / self.step_w
#         x2[0] = x2[0] / self.step_h
#         x2[1] = x2[1] / self.step_w
#         dist_tl = self.__dist__(np.array([np.floor(x1[0]), np.floor(x1[1])]), x2)
#         dist_tr = self.__dist__(np.array([np.floor(x1[0]), np.ceil(x1[1])]), x2)
#         dist_bl = self.__dist__(np.array([np.ceil(x1[0]), np.floor(x1[1])]), x2)
#         dist_br = self.__dist__(np.array([np.ceil(x1[0]), np.ceil(x1[1])]), x2)
#         delta_x = np.modf(x1[1])[0]
#         delta_y = np.modf(x1[0])[0]
#         dist_t = (1 - delta_x) * dist_tl + delta_x * dist_tr
#         dist_b = (1 - delta_x) * dist_bl + delta_x * dist_br
#         dist_x = (1 - delta_y) * dist_t + delta_y * dist_b
#         return dist_x


# PATH = "../inpaint/" + "data/63/63tr.tif"
PATH = "data/63/63tr.tif"
if __name__ == "__main__":
    world_map = np.zeros([4, 4])
    grid_map = np.arange(8).reshape([2, 2, 2])
