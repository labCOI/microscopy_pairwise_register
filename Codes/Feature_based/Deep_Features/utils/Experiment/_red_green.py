import numpy as np


def red_and_green(img1, img2):
    img_red = np.zeros([*img1.shape, 3], np.uint8)
    img_green = np.zeros([*img2.shape, 3], np.uint8)
    img_red[:, :, 2] = img1
    img_green[:, :, 0] = img2
    img_green[:, :, 1] = img2
    return img_red + img_green

