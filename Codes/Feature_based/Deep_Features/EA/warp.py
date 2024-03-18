import ctypes
import os
import sys
import numpy as np
from time import process_time

cdll = ctypes.CDLL(os.path.join(os.path.dirname(os.path.dirname(__file__)), "lib",
                                "warp_win.dll" if sys.platform.startswith("win") else "libwarp.so"))

_warp_uint8 = cdll.warp_uint8
_warp_uint8.restype = None

_warp_float = cdll.warp_float
_warp_float.restype = None

_warp_uint8_mp = cdll.warp_uint8_mp
_warp_uint8_mp.restype = None

_warp_float_mp = cdll.warp_float_mp
_warp_float_mp.restype = None

_affine_uint8 = cdll.affine_uint8
_affine_uint8.restype = None

_affine_uint8_mp = cdll.affine_uint8_mp
_affine_uint8_mp.restype = None
PADDING_ZERO = -1
PADDING_BOUND = -2


def warp_c(img, field, padding_type=0, multi_processor=True):
    assert img.dtype in [np.uint8, np.float32], 'Error Input Type: need float32 or uint8!'
    field = field.astype(np.float32)
    in_type = img.dtype
    height, width = img.shape
    height_field, width_field = field.shape[1:]
    if in_type == np.uint8:
        c_inout_p = ctypes.POINTER(ctypes.c_uint8)
        out_n = ctypes.c_uint8 * (height * width)
    else:
        c_inout_p = ctypes.POINTER(ctypes.c_float)
        out_n = ctypes.c_float * (height * width)
    c_float_p = ctypes.POINTER(ctypes.c_float)
    img = img.reshape(-1)
    field = field.reshape(-1)
    img = img.ctypes.data_as(c_inout_p)
    field = field.ctypes.data_as(c_float_p)
    out = out_n()
    if in_type == np.uint8:
        if multi_processor:
            _warp_uint8_mp.argtypes = [c_inout_p, c_float_p, out_n, ctypes.c_long, ctypes.c_long, ctypes.c_long,
                                       ctypes.c_long, ctypes.c_int]
            if padding_type == 'bound':
                _warp_uint8_mp(img, field, out, height, width, height_field, width_field, PADDING_BOUND)
            else:
                _warp_uint8_mp(img, field, out, height, width, height_field, width_field, PADDING_ZERO)
        else:
            _warp_uint8.argtypes = [c_inout_p, c_float_p, out_n, ctypes.c_long, ctypes.c_long, ctypes.c_long,
                                    ctypes.c_long, ctypes.c_int]
            if padding_type == 'bound':
                _warp_uint8(img, field, out, height, width, height_field, width_field, PADDING_BOUND)
            else:
                _warp_uint8(img, field, out, height, width, height_field, width_field, PADDING_ZERO)
    else:
        if multi_processor:
            _warp_float_mp.argtypes = [c_inout_p, c_float_p, out_n, ctypes.c_long, ctypes.c_long, ctypes.c_long,
                                       ctypes.c_long, ctypes.c_int]
            if padding_type == 'bound':
                _warp_float(img, field, out, height, width, height_field, width_field, PADDING_BOUND)
            else:
                _warp_float(img, field, out, height, width, height_field, width_field, PADDING_ZERO)
        else:
            _warp_float.argtypes = [c_inout_p, c_float_p, out_n, ctypes.c_long, ctypes.c_long, ctypes.c_long,
                                    ctypes.c_long, ctypes.c_int]
            if padding_type == 'bound':
                _warp_float(img, field, out, height, width, height_field, width_field, PADDING_BOUND)
            else:
                _warp_float(img, field, out, height, width, height_field, width_field, PADDING_ZERO)
    out = np.array(out, dtype=in_type).reshape([height, width])
    return out


def affine_c(img, affine,padding_type=0, multi_processor=True):
    assert img.dtype in [np.uint8], 'Error Input Type: need uint8!'
    assert affine.shape == (2, 3), 'Please Input Correct Affine Mat'
    in_type = img.dtype
    height, width = img.shape
    height_field, width_field = img.shape

    c_inout_p = ctypes.POINTER(ctypes.c_uint8)
    out_n = ctypes.c_uint8 * (height * width)

    c_float_p = ctypes.POINTER(ctypes.c_float)
    img = img.reshape(-1)
    affine = affine.reshape(-1)
    img = img.ctypes.data_as(c_inout_p)
    affine = affine.ctypes.data_as(c_float_p)
    out = out_n()
    if multi_processor:
        _affine_uint8_mp.argtypes = [c_inout_p, c_float_p, out_n, ctypes.c_long, ctypes.c_long, ctypes.c_long,
                                     ctypes.c_long, ctypes.c_int]
        if padding_type == 'bound':
            _affine_uint8_mp(img, affine, out, height, width, height_field, width_field, PADDING_BOUND)
        else:
            _affine_uint8_mp(img, affine, out, height, width, height_field, width_field, PADDING_ZERO)
    else:
        _affine_uint8.argtypes = [c_inout_p, c_float_p, out_n, ctypes.c_long, ctypes.c_long, ctypes.c_long,
                                  ctypes.c_long, ctypes.c_int]
        if padding_type == 'bound':
            _affine_uint8(img, affine, out, height, width, height_field, width_field, PADDING_BOUND)
        else:
            _affine_uint8(img, affine, out, height, width, height_field, width_field, PADDING_ZERO)
    out = np.array(out, dtype=in_type).reshape([height, width])
    return out


if __name__ == '__main__':
    import torch

    grid1 = torch.load("grid1.pt")[1]
    grid2 = torch.load("grid2.pt")
    t1 = process_time()
    out1 = warp_c(grid1, grid2)
    t2 = process_time()
    print(t2 - t1)
    t1 = process_time()
    t2 = process_time()
    print(t2 - t1)
