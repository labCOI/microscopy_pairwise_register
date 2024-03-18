import numpy as np
import torch

from time import process_time as ptime

i = 0


def p_inverse(s_mat):
    # return s_mat.transpose(-1, -2) @ torch.inverse(s_mat @ s_mat.transpose(-1, -2))
    return torch.pinverse(s_mat)


def get_affine_mat(from_, to_, device="cuda"):
    src = from_.reshape([-1, 2]).float().to(device)
    dst = to_.reshape([-1, 2]).float().to(device)
    # 求仿射变换矩阵，超过三个点就最小二乘
    # M*S = D->M=D*pinv(S)
    src_3 = torch.cat([src, torch.ones([len(src), 1], device=device)], dim=1).t()
    dst = dst.t()
    m = dst @ p_inverse(src_3)
    return m


def compute_affine(M, from_, device="cuda"):
    src = from_.reshape([-1, 2]).float().to(device)
    src_3 = torch.cat([src, torch.ones([len(src), 1], device=device)], dim=1).t()
    return (M @ src_3).t()


def fine_affine(from_, to_, inliers, rrt=3, device="cuda"):
    inliers_num_pre = inliers.sum()
    m = get_affine_mat(from_[inliers], to_[inliers], device)
    inliers = torch.norm(compute_affine(m, from_, device) - to_, dim=1) < rrt
    inliers_num = inliers.sum()
    while inliers_num > inliers_num_pre:
        inliers_num_pre = inliers_num
        m = get_affine_mat(from_[inliers], to_[inliers], device)
        inliers = torch.norm(compute_affine(m, from_, device) - to_, dim=1) < rrt
        inliers_num = inliers.sum()
    return m, inliers


def ransac(from_, to_, max_iter=512, rrt=3, device="cuda"):
    """
    :param from_:(torch.Tensor)
    :param to_:(torch.Tensor)
    :param max_iter:
    :param rrt: 投影误差
    :param device: 运算设备
    :return: mat: 仿射矩阵 (torch.Tensor)
             inliers: 内点序列 (torch.Tensor)
    """
    src = from_.squeeze().reshape([-1, 2]).float().to(device)
    dst = to_.squeeze().reshape([-1, 2]).float().to(device)
    length = len(src)
    if length < 3:
        return torch.tensor([[1., 0., 0.],
                             [0., 1., 0.]], device=src.device), torch.ones(length, device=device) > 0.5
    torch.random.manual_seed(0)
    idx = []
    for i in range(max_iter):
        idx.append(np.random.choice(np.arange(length), 3, replace=False))
    idx = torch.tensor(idx, device=device, dtype=torch.long)  # (m,3)
    src_3 = torch.cat([src, torch.ones([len(src), 1], device=device)], dim=1).t()  # (n,2)->(3,n)
    src_sample = src_3[:, idx.reshape(-1)].reshape([-1, *idx.shape]).permute([1, 0, 2])  # (m,x,i) m:maxiter x:(x,y,1),
    # i:三个坐标
    dst_sample = dst[idx.reshape(-1), :].reshape([*idx.shape, -1]).permute([0, 2, 1])  # (m, x, i)
    m = dst_sample @ p_inverse(src_sample)
    # 验证
    src_ver = src_3.unsqueeze(0).repeat([max_iter, 1, 1])
    dst_var = dst.unsqueeze(0).repeat([max_iter, 1, 1]).permute([0, 2, 1])
    err = torch.norm((m @ src_ver - dst_var), dim=1) < rrt
    m_idx = torch.argmax(err.sum(dim=1))
    return fine_affine(from_, to_, err[m_idx], 3, device=device)


if __name__ == "__main__":
    device = "cuda"
