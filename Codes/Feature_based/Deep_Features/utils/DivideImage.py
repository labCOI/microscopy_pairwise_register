import torch
import os
import math
import numpy as np
import cv2
from torch.utils.data.dataset import Dataset


def merge_pair(img_base, img_patch):
    overlap_mask = (abs(img_base) > 1e-6) & (abs(img_patch) > 1e-6)
    img_rst = overlap_mask * (0.5 * img_base + 0.5 * img_patch) + (~overlap_mask) * (img_base + img_patch)
    return img_rst


def merge_serial(img_lst):
    """
    :param img_lst: 需要合在一起的图像
    :return: 返回和原来一样格式的图像
    """
    if isinstance(img_lst[0], torch.Tensor):
        img_tor = torch.cat(img_lst)
        # img_np = np.array(img_tor)
        overlap_weight = 1 / ((abs(img_tor) > 1e-6).float()).sum(0)
        img_rst = (overlap_weight * img_tor).sum(0)
    return img_rst
    # img_rst = img_lst[0]
    # for img in img_lst:
    #     img_rst = merge_pair(img_rst, img)
    # return img_rst


class DivideImage:
    """
    将一张图分成好多块然后每块分别计算
    """

    def __init__(self, img_org, patch_sz=512, overlap=0, mask=None, scale=1):
        """
        :param img_org: 原始图像
        :param patch_sz: 每块大小
        :param overlap: 重叠比率0-0.5, 若大于1则表示像素
        :param mask: 用来标识不需要做操作的地方
        :param scale: 缩放

        Example:
        img_d = DivideImage(img(numpy), patch=1024)
        for i in range(len(img_d)):
            img, idxs = img_d[i]
            do something
        """
        super(DivideImage, self).__init__()
        # 参数
        self.img = img_org
        self.changed = np.zeros(self.img.shape)
        self.mask = mask
        self.patch_sz = patch_sz
        self.overlap = min(overlap if overlap > 0.5 else int(overlap * patch_sz), patch_sz)
        self.scale = scale
        # 属性
        self.height = img_org.shape[0]
        self.width = img_org.shape[1]
        # 保存数据， 用左上角和右下角两个点来标记图像
        self.__patches_org = []
        self.__idx_org = []
        self.__patches_changed = []
        self.__idx_changed = []
        # 生成patch
        if patch_sz > min(self.height, self.width):
            self.patch_sz = min(self.height, self.width)
        self.__generate_patches()

    def __generate_patches(self):
        """
        生成patches
        :return:
        """
        step = int((self.patch_sz - self.overlap) // self.scale)
        lap_len = int(self.patch_sz // self.scale) - step
        for i in range(math.ceil((self.height - lap_len) / step)):
            for j in range(math.ceil((self.width - lap_len) / step)):
                if i == math.ceil((self.height - lap_len) / step) - 1:
                    idx1 = self.height - int(self.patch_sz // self.scale)
                    idx2 = self.height
                else:
                    idx1 = i * step
                    idx2 = idx1 + int(self.patch_sz // self.scale)
                if j == math.ceil((self.width - lap_len) / step) - 1:
                    idy1 = self.width - int(self.patch_sz // self.scale)
                    idy2 = self.width
                else:
                    idy1 = j * step
                    idy2 = idy1 + int(self.patch_sz // self.scale)
                if self.mask is None or self.mask[idx1:idx2, idy1:idy2].max() > 0:
                    img = cv2.resize(self.img[idx1:idx2, idy1:idy2], (self.patch_sz, self.patch_sz))
                    self.__patches_org.append(img)
                    self.__idx_org.append([idx1, idx2, idy1, idy2])

    def __merge(self, patches, indexes):
        """
        将修改后的输出合并起来
        :param patches:
        :param indexes:
        :param mask:
        :return:
        """
        for i, (patch, idx) in enumerate(zip(patches, indexes)):
            idx1 = idx[0]
            idx2 = idx[1]
            idy1 = idx[2]
            idy2 = idx[3]
            patch = cv2.resize(patch, (int(self.patch_sz // self.scale), int(self.patch_sz // self.scale)))
            self.changed[idx1:idx2, idy1:idy2] = merge_pair(self.changed[idx1:idx2, idy1:idy2], patch)

    def add_changed(self, patch, idx):
        self.__patches_changed.append(patch)
        self.__idx_changed.append(idx)

    def merged_output(self):
        """
        返回修改后的图像
        :return:
        """
        if self.changed is not []:
            self.__merge(self.__patches_changed, self.__idx_changed)
        return self.changed

    def __getitem__(self, item):
        return self.__patches_org[item], self.__idx_org[item]

    def __len__(self):
        return len(self.__patches_org)


class BigImage(Dataset):
    def __init__(self, img, *args, **kwargs):
        """
        单独一张大图
        """
        super(BigImage, self).__init__()
        # 存储图像
        assert isinstance(img, np.ndarray) or (
                isinstance(img, str) and os.path.isfile(img)), "Error input:please input file/dir or ndarry"
        if isinstance(img, np.ndarray):
            self.img = DivideImage(img, *args, **kwargs)
        else:
            self.img = DivideImage(cv2.imread(img, cv2.IMREAD_GRAYSCALE), *args, **kwargs)

    def __getitem__(self, item):
        img, idx = self.img[item]
        img = img.reshape([-1, img.shape[-2], img.shape[-1]])
        idx = torch.tensor(idx)
        return (torch.tensor(img).type(torch.float32) / 255), idx

    def add_item(self, patch, idx):
        patch = np.array(patch)
        idx = idx.numpy()
        for p, i in zip(patch, idx):
            self.img.add_changed(p.squeeze(), i)

    def merge(self):
        return self.img.merged_output()

    def __len__(self):
        return len(self.img)


class BigImageSet(Dataset):
    def __init__(self, img, *args, **kwargs):
        """
        供dataloader使用的类, 多线程可能会出错, 并且不支持shuffle
        :param img: 图像可能是ndarray图像,ndarray_list,文件夹,文件夹list,文件,文件list
        :param patch_sz: 每块大小
        :param overlap: 重叠比率0-0.5, 若大于1则表示像素
        :param mask: 用来标识不需要做操作的地方
        :param scale: 缩放
        """
        super(BigImageSet, self).__init__()
        # 处理不同类型的数据
        if isinstance(img, np.ndarray) or isinstance(img, str):
            img = [img]
        assert isinstance(img, list), "Error input:please input file/dir or ndarry"
        # 参数
        self.img = img
        self.arg = args
        self.kwarg = kwargs
        # 存储

    def merge(self):
        pass

    def __getitem__(self, item):
        pass

    def __len__(self):
        pass
