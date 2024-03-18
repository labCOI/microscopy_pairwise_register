# -*- coding: utf-8 -*-
# @Time    : 2021/7/1 20:10
# @Author  : XinTong
# @FileName: _show_cross_section.py
# @Software: PyCharm
import skimage.io as io
import skimage
import numpy as np
import random
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('-d', '--Data', default='data/UNet_AE_serial.tif')
parser.add_argument('-l', '--Label', default='data/UNet_AE_lserial.tif')
parser.add_argument('-x', '--Xaxis', default=None, type=int)
parser.add_argument('-y', '--Yaxis', default=None, type=int)
parser.add_argument('-f', '--Factor', default=5, type=int)
parser.add_argument('-o', '--Out', default='test.tif')

color_code = np.array([0, 36, 221, 188, 68, 104, 214, 19, 239, 144, 169, 96, 240, 46, 205, 113, 25,  133, 200, 44, 8, 204, 253, 6, 196, 148, 201, 216, 147, 53, 224, 194, 152, 21, 62, 134, 17, 199, 27, 67, 64, 238, 22, 84, 130, 117, 90, 162, 14, 250, 226, 126, 54, 177, 35, 119, 128, 228, 163, 161, 72, 248, 39, 241, 132, 223, 178, 88, 142, 234, 106, 237, 146, 252, 47, 155, 11, 98, 34, 41, 206, 123, 30, 192, 127, 108, 118, 181, 159, 145, 233, 107, 12, 3, 55, 20, 103, 57, 105, 24, 164, 95, 66, 76, 2, 157, 249, 70, 37, 154, 43, 232, 91, 203, 102, 52, 99, 80, 180, 243, 166, 83, 18, 114, 121, 4, 179, 116, 10, 51, 69, 32, 137, 184, 191, 58, 75, 170, 150, 202, 198, 78, 87, 227, 222, 229, 120, 124, 71, 49, 7, 230, 5, 73, 77, 115, 82, 110, 138, 81, 38, 143, 176, 65, 13, 149, 136, 94, 158, 208, 100, 182, 173, 165, 215, 197, 23, 183, 189, 217, 9, 172, 212, 129, 139, 244, 236, 29, 219, 56, 242, 122, 235, 16, 193, 171, 74, 26, 255, 210, 220, 175, 79, 101, 31, 174, 40, 61, 213, 15, 168, 207, 42, 28, 209, 141, 60, 246, 1, 63, 195, 218, 231, 125, 85, 33, 50, 131, 59, 225, 186, 247, 251, 97, 254, 45, 111, 153, 187, 211, 140, 151, 48, 135, 156, 92, 89, 112, 93, 86, 185, 167, 190, 245, 160, 109]
)


def colored(im_o, alpha=1.):
    im = im_o[:, :, np.newaxis]
    h = color_code[im] / 255.
    s = im > 0
    v = (im > 0) * alpha
    hsv = np.concatenate([h, s, v], axis=2)
    colored_im = skimage.color.hsv2rgb(hsv) * 255
    return colored_im.astype(np.uint8)


def cross_section_vertical(im_o, x=None, y=None, times=1, label=None, alpha=0.3, colors=None):
    """
    :param im_o: 图像stack(numpy.array(N x h x w))
    :param x: 从x处垂直于y切一刀，优先
    :param y: 从y处垂直于x切一刀
    :param times: z分辨率与xy轴分辨率的
    :param label: 用来标记颜色
    :param alpha: 颜色深度
    :return: im: 横截面图像(numpy.array(h x w))
    Example:
    img = numpy.array([N,w,h])
    img = cross_section_vertical(img, x=x, y=None, times = factor, label=label)
    """
    if label is None:
        label = np.zeros_like(im_o)
    if x is None and y is None:
        im = im_o[0, :, :]
        lb = label[0, :, :]
    else:
        if x is not None:
            im = im_o[:, x, :]#.repeat(times, axis=0)
            lb = label[:, x, :]#.repeat(times, axis=0)
        elif y is not None:
            im = im_o[:, :, y]#.repeat(times, axis=0)
            lb = label[:, :, y]#.repeat(times, axis=0)
        im = (skimage.transform.resize(im, [int(im.shape[0]*times), im.shape[1]]) * 255).astype(np.uint8)
        lb = (skimage.transform.resize(lb, [int(lb.shape[0] * times), lb.shape[1]], order=0) *255).astype(np.uint8)
        im[lb>0] = im[lb>0] * (1-alpha)
    if colors is None:
        colored_label = colored(lb, alpha=alpha)
    else:
        colored_label = np.zeros_like(lb)[:,:,np.newaxis].repeat(3, axis=-1)
        lb_idx = np.unique(lb)
        for i in lb_idx[1:]:
            colored_label[lb == i,:] = colors[i-1]
        colored_label = (colored_label*alpha).astype(np.uint8)
    im = im[:, :, np.newaxis].repeat(3, axis=-1)  + colored_label
    return im


if __name__ == "__main__":
    args = parser.parse_args()
    DATA = args.Data
    LABEL = args.Label
    X = args.Xaxis
    Y = args.Yaxis
    FACTOR = args.Factor
    OUT = args.Out
    img = io.imread(DATA)
    if LABEL is not None:
        label = io.imread(LABEL)
        img = cross_section_vertical(img, x=X, y=Y, times=FACTOR, label=label)
    else:
        img = cross_section_vertical(img, x=X, y=Y, times=FACTOR)
    io.imsave(OUT, img)
