import torch
import cv2
import numpy as np
import PIL.ImageFile
import PIL.Image as Image


class Convert2:
    def __init__(self, im):
        if isinstance(im, torch.Tensor):
            im = im.detach().cpu()
        self.im = np.array(im)

    def convert2(self, c):
        if c in ['Torch', 'torch', 'Tensor', 'tensor', 'pytorch', 'Pytorch']:
            self.im = torch.from_numpy(self.im)
        elif c in ['PIL', 'Image']:
            self.im = Image.fromarray(self.im)
        return self.im

    def float(self):
        self.im = self.im.astype(np.float32)

    def uint8(self):
        self.im = self.im.astype(np.uint8)

    # 应该只压缩维度
    def set_dim(self, dim):
        self.im = np.squeeze(self.im)
        for _ in range(dim - self.im.ndim):
            self.im = self.im[np.newaxis, :]

    def alpha(self, alpha):
        self.im = self.im * alpha


def Any2Torch(im, dim=4):
    if isinstance(im, torch.Tensor):
        return im
    c = Convert2(im)
    if isinstance(im, PIL.ImageFile.ImageFile):
        c.float()
        c.alpha(1 / 255.)
    elif c.im.dtype == np.uint8:
        c.float()
        c.alpha(1 / 255.)
    c.set_dim(dim)
    if dim is 4:
        if c.im.shape[-1] is 3:
            c.im = c.im.transpose([0, 3, 1, 2])
    else:
        if c.im.shape[-1] is 3:
            c.im = c.im.transpose([2, 0, 1])
    return c.convert2('Torch')


def Any2PIL(im):
    if isinstance(im, PIL.ImageFile.ImageFile):
        return im
    c = Convert2(im)
    if isinstance(im, torch.Tensor):
        c.alpha(255.)
    c.uint8()
    c.set_dim(2)
    if c.im.ndim is 3 and c.im.shape[0] == 3:
        c.im = c.im.transpose([1, 2, 0])
    return c.convert2('PIL')


def Any2np(im):
    c = Convert2(im)
    return c.convert2('np')


def kpl2lst(kpl):
    """
    将cv2的KeyPoint list转换成[Nx2]的numpy
    :param kpl:KeyPoint list
    :return:
    """
    if isinstance(kpl, list):
        return np.array([pt.pt for pt in kpl]).astype(np.float32)
    else:
        return kpl.squeeze()


def lst2kpl(lst):
    """
    将numpy表示的点转回KeyPoint list
    :param lst:
    :return: list<KeyPoint>
    """
    return [cv2.KeyPoint(float(pt[0]), float(pt[1]), 32) for pt in lst]


# --------------------------------------------------------------------------
# 转换仿射矩阵
# --------------------------------------------------------------------------
def get_N(W, H):
    """N that maps from unnormalized to normalized coordinates"""
    N = np.zeros((3, 3), dtype=np.float64)
    N[0, 0] = 2.0 / W
    N[0, 1] = 0
    N[1, 1] = 2.0 / H
    N[1, 0] = 0
    N[0, -1] = -1.0
    N[1, -1] = -1.0
    N[-1, -1] = 1.0
    return N


def get_N_inv(W, H):
    """N that maps from normalized to unnormalized coordinates"""
    N = get_N(W, H)
    return np.linalg.inv(N)


def cv_m2theta(M, w, h):
    """convert affine warp matrix `M` compatible with `opencv.warpAffine` to `theta` matrix
    compatible with `torch.F.affine_grid`

    Parameters
    ----------
    M : np.ndarray
        affine warp matrix shaped [2, 3]
    w : int
        width of image
    h : int
        height of image

    Returns
    -------
    np.ndarray
        theta tensor for `torch.F.affine_grid`, shaped [2, 3]
    """
    M_aug = np.concatenate([M, np.zeros((1, 3))], axis=0)
    M_aug[-1, -1] = 1.0
    N = get_N(w, h)
    N_inv = get_N_inv(w, h)
    theta = N @ M_aug @ N_inv
    theta = np.linalg.inv(theta)
    return theta[:2, :]


def theta2cv_m(theta, w, h, return_inv=False):
    """convert theta matrix compatible with `torch.F.affine_grid` to affine warp matrix `M`
    compatible with `opencv.warpAffine`.

    Note:
    M works with `opencv.warpAffine`.
    To transform a set of bounding box corner points using `opencv.perspectiveTransform`, M^-1 is required

    Parameters
    ----------
    theta : np.ndarray
        theta tensor for `torch.F.affine_grid`, shaped [2, 3]
    w : int
        width of image
    h : int
        height of image
    return_inv : False
        return M^-1 instead of M.

    Returns
    -------
    np.ndarray
        affine warp matrix `M` shaped [2, 3]
    """
    theta_aug = np.concatenate([theta, np.zeros((1, 3))], axis=0)
    theta_aug[-1, -1] = 1.0
    N = get_N(w, h)
    N_inv = get_N_inv(w, h)
    M = np.linalg.inv(theta_aug)
    M = N_inv @ M @ N
    if return_inv:
        M_inv = np.linalg.inv(M)
        return M_inv[:2, :]
    return M[:2, :]
