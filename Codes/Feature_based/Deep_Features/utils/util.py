import os
import sys
import cv2
import ctypes
import numpy as np
import subprocess

cdll = ctypes.CDLL(os.path.join(os.path.dirname(os.path.dirname(__file__)), "lib",
                                "pyr_read.dll" if sys.platform.startswith("win") else "libpyr_read.so"))
_layer_num = cdll.layer_num
_layer_num.restype = ctypes.c_int32

_img_size = cdll.img_size
_img_size.restype = None

_get_img = cdll.get_img
_get_img.restype = None

_get_rect = cdll.get_rect
_get_rect.restype = None


def layer_num(file_name):
    file_name_utf8 = file_name.encode(encoding="utf-8")
    _layer_num.argtypes = [ctypes.c_char_p, ctypes.c_int]
    _layer_num(file_name_utf8, len(file_name))
    return _layer_num(file_name_utf8, len(file_name))


def img_size(file_name, scale2=1):
    file_name_utf8 = file_name.encode(encoding="utf-8")
    c_long2 = ctypes.c_long * 2
    size = c_long2()
    _img_size.argtypes = [ctypes.c_char_p, ctypes.c_int, ctypes.c_int, c_long2]
    _img_size(file_name_utf8, len(file_name), scale2, size)
    size = np.array(size)
    return size


def save_mosaic(dir_name, img, img_sz=12288):
    """
    写mosaic2
    :param dir_name:
    :param img:
    :param img_sz:
    :return:
    """
    height, width = img.shape
    os.makedirs(os.path.join(dir_name, 'data'), exist_ok=True)
    with open(os.path.join(dir_name, "index.txt"), mode="w+") as f:
        height_idx = int(np.ceil(height // img_sz)) + 1
        width_idx = int(np.ceil(width // img_sz)) + 1
        out_line = str(height_idx) + "," + str(width_idx) + "," + str(img_sz) + "," + str(img_sz) + "\n"
        f.write(out_line)
        for j in range(height_idx):
            for i in range(width_idx):
                y_top = j * img_sz
                x_left = i * img_sz
                y_bottom = (j + 1) * img_sz
                x_right = (i + 1) * img_sz
                if x_right > width or y_bottom > height:
                    y_bottom = min(y_bottom, height)
                    x_right = min(x_right, width)
                    img_tmp = np.zeros([img_sz, img_sz], dtype=np.uint8)
                    img_tmp[:y_bottom - y_top, :x_right - x_left] = img[y_top:y_bottom, x_left:x_right]
                else:
                    img_tmp = img[y_top:y_bottom, x_left:x_right]
                if cv2.imwrite(os.path.join(dir_name, str(j + 1) + '_' + str(i + 1) + ".bmp"), img_tmp):
                    out_line = str(j + 1) + "," + str(i + 1) + "," + str(x_left) + "," + str(y_top) + "\n"
                    f.write(out_line)


def read_pyr(file_name, scale_2=1):
    """
    读金字塔文件
    :param file_name:
    :param scale_2: 向下缩放2^(scale2-1)倍
    :return:
    """
    scale_2 = int(max(min(layer_num(file_name), scale_2), 1))
    size_org = img_size(file_name, 1)
    size = img_size(file_name, scale_2)
    file_name_utf8 = file_name.encode(encoding="utf-8")
    c_img_type = ctypes.c_uint8 * int(size[0]) * int(size[1])
    img = c_img_type()
    _get_img.argtypes = [ctypes.c_char_p, ctypes.c_int, ctypes.c_int, c_img_type]
    _get_img(file_name_utf8, len(file_name), scale_2, img)
    img = np.array(img).reshape(size)

    height = size_org[0] // (2 ** (scale_2 - 1))
    width = size_org[1] // (2 ** (scale_2 - 1))
    img = img[:height, :width]
    if img.shape[0] > height or img.shape[1] > width:
        img = np.pad(img, ((0, img.shape[0] - height), (0, img.shape[1] - width)))
    return img


def roi_pyr(file_name, scale_2, rect):
    """
    读金字塔文件
    :param file_name:
    :param rect: roi区域 (left,top,right,bottom)
    :return:
    """
    rect = np.array(rect).astype(np.long)
    h, w = rect[3] - rect[1], rect[2] - rect[0]
    c_rec_type = ctypes.c_long * 4
    c_img_type = ctypes.c_uint8 * (h * w)
    img = c_img_type()
    rect = c_rec_type.from_buffer(rect)
    file_name_utf8 = file_name.encode(encoding="utf-8")
    _get_rect.argtypes = [ctypes.c_char_p, ctypes.c_int, ctypes.c_int, c_rec_type, c_img_type]
    _get_rect(file_name_utf8, len(file_name), scale_2, rect, img)
    img = np.array(img).reshape([h, w])
    return img


def save_2fiji(dir_name, img, img_num=1, img_sz=4000):
    """
    写2fiji
    :param dir_name:
    :param img:
    :param img_sz:
    :return:
    """
    height, width = img.shape
    os.makedirs(os.path.join(dir_name), exist_ok=True)
    with open(os.path.join(dir_name, "mosaic2fiji.txt"), mode="w+") as f:
        height_idx = int(np.ceil(height // img_sz))
        width_idx = int(np.ceil(width // img_sz))
        for j in range(height_idx):
            for i in range(width_idx):
                y_top = j * img_sz
                x_left = i * img_sz
                y_bottom = (j + 1) * img_sz
                x_right = (i + 1) * img_sz
                if x_right > width or y_bottom > height:
                    y_bottom = min(y_bottom, height)
                    x_right = min(x_right, width)
                    img_tmp = np.zeros([img_sz, img_sz], dtype=np.uint8)
                    img_tmp[:y_bottom - y_top, :x_right - x_left] = img[y_top:y_bottom, x_left:x_right]
                else:
                    img_tmp = img[y_top:y_bottom, x_left:x_right]
                file_name = "layer" + str(img_num) + '_' + str(j + 1) + '_' + str(i + 1) + ".jpg"
                if cv2.imwrite(os.path.join(dir_name, file_name), img_tmp):
                    out_line = file_name + ',' + str(x_left) + ',' + str(y_top) + ',' + str(img_num) + '\n'
                    f.write(out_line)


def read_2fiji(file_name, img_num, scale=1):
    if not os.path.exists(file_name):
        return None
    idx_file = os.path.join(file_name, 'mosaic2fiji.txt')
    with open(idx_file) as f:
        mosaic = f.readlines()
        mosaic_focus = []
        for m in mosaic:
            m = m.strip().split(',')
            if int(m[-1]) == img_num:
                mosaic_focus.append(m)
        f.close()
    if len(mosaic_focus) == 0:
        return None
    # image size of one mosaic
    im = cv2.imread(os.path.join(file_name, mosaic_focus[0][0]), cv2.IMREAD_GRAYSCALE)
    sz = im.shape[0]
    height = int(mosaic_focus[-1][2]) + sz
    width = int(mosaic_focus[-1][1]) + sz
    img = np.zeros([height, width], np.uint8)
    for m in mosaic_focus:
        mosaic_name, idx_w, idx_h, _ = m
        idx_h, idx_w = int(idx_h), int(idx_w)
        img[idx_h:idx_h + sz, idx_w:idx_w + sz] = cv2.imread(os.path.join(file_name, mosaic_name),
                                                             cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, None, fx=scale, fy=scale)
    return img


def roi_2fiji(file_name, img_num, scale, rect):
    """
    读2fiji文件
    :param file_name:
    :param img_num:
    :param scale:
    :param rect: roi区域 (left,top,right,bottom)
    :return:
    """
    # 读2fiji.txt 获得各项属性
    if not os.path.exists(file_name):
        return None
    idx_file = os.path.join(file_name, 'mosaic2fiji.txt')
    with open(idx_file) as f:
        mosaic = f.readlines()
        mosaic_focus = []
        for m in mosaic:
            m = m.strip().split(',')
            if int(m[-1]) == img_num:
                mosaic_focus.append(m)
        f.close()
    if len(mosaic_focus) == 0:
        return None
    # image size of one mosaic
    im = cv2.imread(os.path.join(file_name, mosaic_focus[0][0]), cv2.IMREAD_GRAYSCALE)
    sz = im.shape[0]
    height = int(mosaic_focus[-1][2]) + sz
    width = int(mosaic_focus[-1][1]) + sz
    # rect的长宽
    rect_height, rect_width = rect[3] - rect[1], rect[2] - rect[0]
    rect = np.array(rect)
    mosaic_l, mosaic_t, mosaic_r, mosaic_b = np.mod(rect, sz)
    idx_l, idx_t, idx_r, idx_b = rect // sz
    img = np.zeros([rect_height, rect_width], dtype=np.uint8)
    for j in range(idx_t, idx_b + 1):
        for i in range(idx_l, idx_r + 1):
            img_l, img_t, img_r, img_b = 0 if i == idx_l else (i - idx_l) * sz - mosaic_l, \
                                         0 if j == idx_t else (j - idx_t) * sz - mosaic_t, \
                                         rect_width if i == idx_r else (i - idx_l + 1) * sz - mosaic_l, \
                                         rect_height if j == idx_b else (j - idx_t + 1) * sz - mosaic_t
            roi_l, roi_t, roi_r, roi_b = mosaic_l if i == idx_l else 0, \
                                         mosaic_t if j == idx_t else 0, \
                                         mosaic_r if i == idx_r else sz, \
                                         mosaic_b if j == idx_b else sz
            img_name = mosaic_focus[j * width // sz + i][0]
            img[img_t:img_b, img_l:img_r] = \
                cv2.imread(os.path.join(file_name, img_name), cv2.IMREAD_GRAYSCALE)[roi_t:roi_b, roi_l:roi_r]
    return img


def read_mira(file_name, num=None, data_mode='pyr', scale=1):
    if data_mode == 'pyr':
        scale2 = np.floor(np.log2(1 / scale)).astype(np.int) + 1
        scale_next = scale * (2 ** (scale2 - 1))
        if os.path.isdir(file_name):
            file_name = os.path.join(file_name, 'PyrImg.dat')
        if os.path.exists(file_name):
            if scale_next == 1:
                return read_pyr(file_name, scale2)
            else:
                return cv2.resize(read_pyr(file_name, scale2), None, fx=scale_next, fy=scale_next)
        else:
            return None
    elif data_mode == 'fiji':
        if not isinstance(num, int):
            if num.text() == '':
                return None
            num = int(num.text())
        return read_2fiji(file_name, num, scale)
    else:
        img = cv2.imread(file_name, cv2.IMREAD_GRAYSCALE)
        if img is None:
            return img
        else:
            return cv2.resize(img, None, fx=scale, fy=scale)


def save_mira(dir_name, img, num=None, data_mode='pyr', img_sz=12288):
    if data_mode == 'pyr' or data_mode == "mosaic":
        save_mosaic(dir_name, img, img_sz)
        print("save mosaic finished!")
        if data_mode == 'pyr' and sys.platform.startswith("win"):
            exec_path = "lib/SeqImageMosaicMerge.exe"
            file_name = dir_name + '/'
            index_name = file_name + "index.txt"
            subprocess.Popen([exec_path, file_name, index_name])
    else:
        save_2fiji(dir_name, img, int(num.text()), img_sz)


def read_size_mira(file_name, num=None, data_mode='pyr', scale=1):
    if data_mode == 'pyr':
        scale2 = np.floor(np.log2(1 / scale)).astype(np.int) + 1
        scale_next = scale * (2 ** (scale2 - 1))
        if os.path.isdir(file_name):
            file_name = os.path.join(file_name, 'PyrImg.dat')
        if os.path.exists(file_name):
            if scale_next == 1:
                return img_size(file_name, scale2)
            else:
                return img_size(file_name, scale2) * scale_next
        else:
            return None
    elif data_mode == 'fiji':
        assert "error"
    else:
        img = cv2.imread(file_name, cv2.IMREAD_GRAYSCALE)
        if img is None:
            return img
        else:
            return img.shape * scale


if __name__ == "__main__":
    filename = "/home/xint/TempData/wafer14Data_76224_147904/mosaic2fiji.txt"
