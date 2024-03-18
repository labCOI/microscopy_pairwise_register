from utils.torchsummary import summary
from utils.convert import *
import numpy as np
import cv2
import matplotlib.pyplot as plt
import math


# 将CNN运行过程中的tensor画下来
def save_tensor(path, tensor):
    img = Any2PIL(tensor)
    img.save(path)


def same_size(img1, img2):
    # 长宽都调整到较大值
    h_max = max(img1.shape[0], img2.shape[0])
    w_max = max(img1.shape[1], img2.shape[1])
    img1 = np.pad(img1, [[0, h_max - img1.shape[0]],
                         [0, w_max - img1.shape[1]]])
    img2 = np.pad(img2, [[0, h_max - img2.shape[0]],
                         [0, w_max - img2.shape[1]]])
    return img1, img2


def _convert_pt(pts):
    """
    将点转化为(n,2)的numpy形式
    :param pts:
    :return:
    """
    if pts is None:
        return np.array([])
    if isinstance(pts, list):
        if isinstance(pts[0], cv2.KeyPoint):
            return np.array([pt.pt for pt in pts])
    if isinstance(pts, np.ndarray):
        return pts.reshape([-1, 2])
    if isinstance(pts, torch.Tensor):
        pts = pts.cpu().numpy()
        return pts.reshape([-1, 2])


# 在图像上绘出关键点
def draw_pts(img, pts, color=None, size=1, name="img", radius=1, thickness=2):
    """
    :param color:
    :param img: 绘制在img上
    :param pts: 绘制的点集， opencv的或者numpy(n,2)(1,n,2)都可以
    :param size: 最后显示图片的大小
    :param name: 显示窗口的名字
    :param radius: 半径
    :param thickness: 线宽
    :return:img 返回绘制好的图像
    Example:
    img = draw_pts(img,pts)
    """
    pts = _convert_pt(pts)
    if img.shape[-1] != 3:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    for point in pts:
        if color is None:
            color = (np.random.randint(0, 255), np.random.randint(0, 255), np.random.randint(0, 255))
        img = cv2.circle(img, (int(point[0]), int(point[1])), radius=radius, color=color, thickness=thickness)
    if isinstance(size, int):
        size = float(size)
    if isinstance(size, float):
        img = cv2.resize(img, (int(size * img.shape[1]), int(size * img.shape[0])))
    else:
        img = cv2.resize(img, (int(size[1]), int(size[0])))
    # cv2.imshow(name, img)
    if name is not None:
        plt.figure(name)
        plt.imshow(img)
        plt.show()
    return img


def draw_arrows(img, pts1, pts2, name=None, color=None, line_width=2, tip_len=0.5, inlier=None, ):
    """
    绘制一批箭头
    :param img:
    :param pts1: 从pt1出发
    :param pts2: 指向pt2
    :param name:
    :param color:
    :param line_width: 线的长度
    :param tip_len: 箭头头的长度
    :param inlier: 是否是内点
    :return: img 返回绘制好的图像
    """
    pts1 = _convert_pt(pts1)
    pts2 = _convert_pt(pts2)
    if img.shape[-1] != 3:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    for i, (point1, point2) in enumerate(zip(pts1, pts2)):
        if inlier is not None and inlier[i] == 0:
            continue
        if color is None:
            c = (np.random.randint(0, 255), np.random.randint(0, 255), np.random.randint(0, 255))
            img = cv2.arrowedLine(img, tuple(point1), tuple(point2), c, thickness=line_width, tipLength=tip_len)
        else:
            img = cv2.arrowedLine(img, tuple(point1), tuple(point2), color, thickness=line_width, tipLength=tip_len)
    if name is not None:
        img_sz = cv2.resize(img, (1600, 1600))
        plt.figure(name)
        plt.imshow(img_sz)
        plt.show()
    return img


def draw_correspond(img1, img2, pts1, pts2, size=1, color=None, pt_sz=2, line_width=2, name='img', inlier=None):
    """
    绘制两张图片的匹配关系，两张图的大小需要一致
    :param img1:图1
    :param img2:图2
    :param pts1:图1的点， opencv的或者numpy(n,2)(1,n,2)都可以
    :param pts2:图2的点
    :param size:最终的图像大小的1/2
    :param color:匹配颜色
    :param pt_sz:点的大小
    :param line_width:线的宽度
    :param name:绘制窗口的名字， None为不显示绘制窗口
    :param inlier:内点集合，排除掉一些不想要的点
    :return: img 返回绘制好的图像

    Example:
    img = draw_pts2(img1,img2,pts1,pts2)
    """
    pts1 = _convert_pt(pts1)
    pts2 = _convert_pt(pts2)
    if img1.shape[-1] != 3:
        img1 = cv2.cvtColor(img1, cv2.COLOR_GRAY2BGR)
    if img2.shape[-1] != 3:
        img2 = cv2.cvtColor(img2, cv2.COLOR_GRAY2BGR)
    img = np.concatenate([img1, img2], axis=1)
    width = img1.shape[1]
    if color is None:
        color = (np.random.randint(0, 255), np.random.randint(0, 255), np.random.randint(0, 255))
    for i, (point1, point2) in enumerate(zip(pts1, pts2)):
        if inlier is not None and inlier[i] == 0:
            continue
        if isinstance(color, list):
            co = color[i]
        else:
            co = color
        point1 = (int(point1[0]), int(point1[1]))
        point2 = (int(point2[0]) + width, int(point2[1]))
        if pt_sz !=0:
            img = cv2.circle(img, tuple(point1), pt_sz, co, -1)
            img = cv2.circle(img, tuple(point2), pt_sz, co, -1)
        if line_width != 0:
            img = cv2.line(img, tuple(point1), tuple(point2), co, line_width)
    if isinstance(size, int):
        size = float(size)
    if isinstance(size, float):
        img = cv2.resize(img, (int(2 * size * img1.shape[1]), int(size * img1.shape[0])))
    else:
        img = cv2.resize(img, (int(2 * size[1]), int(size[0])))
    if name is not None:
        plt.figure(name)
        plt.imshow(img)
        plt.show()
    return img


def draw_grid(img=None, sz=None, color=(0, 255, 0), line_width=1, density=0.02):
    """
    画出可以在上面进行形变的网格图

    :param img: 在图上画网格 numpy[N,N] or [N,N,3]
    :param sz: 按照一定的大小画网格 [X,Y]
    :param color: 网格线的颜色，默认绿色
    :param line_width: 网格线的长度
    :param density: 网格线密度
    :return: 网格图像
    """
    assert img is not None or sz is not None, "img or sz must have one decided"
    if density < 1:
        density = int(1 / density)
    if img is not None:
        assert isinstance(img, np.ndarray), "the img must be ndarray"
        if len(img.shape) == 2:
            img = cv2.cvtColor(img.squeeze(), cv2.COLOR_GRAY2BGR)
            sz = img.shape
    else:
        img = np.zeros([sz[0], sz[1], 3]).astype(np.uint8)
        sz = img.shape

    height, width = sz[:2]
    for i in range(0, width, density):
        img = cv2.line(img, (0, i), (height, i), color=color, thickness=line_width)
    for j in range(0, height, density):
        img = cv2.line(img, (j, 0), (j, width), color=color, thickness=line_width)
    return img


def combine_pic(img_lst, w_num, margin=0, scale=1):
    """
    将图片拼成一张显示
    :param img_lst:图像列表
    :param w_num: 一行有几张图片
    :param margin: 图片拼接的间隔
    :param scale: 缩放
    :return:
    """
    height = int(img_lst[0].shape[0] * scale)
    width = int(img_lst[0].shape[1] * scale)
    num = len(img_lst)
    h_num = math.ceil(num / w_num)
    img_rst = np.zeros([h_num * (height + margin) - margin, w_num * (width + margin) - margin, 3], dtype=np.uint8)
    for i, img in enumerate(img_lst):
        y_idx = int(i / w_num)
        x_idx = (i % w_num)
        l = x_idx * (width + margin)
        r = l + width
        t = y_idx * (height + margin)
        b = t + height
        img_rst[t:b, l:r, :] = cv2.resize(img, None, fx=scale, fy=scale)
    return img_rst
