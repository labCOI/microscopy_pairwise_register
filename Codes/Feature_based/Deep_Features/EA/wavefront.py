import os
import sys
import numpy as np
import ctypes
from time import perf_counter

# if sys.platform.startswith("win"):
if False:
    from EA.WaveFront import wavefront
else:
    cdll = ctypes.CDLL(os.path.join(os.path.dirname(os.path.dirname(__file__)), "lib",
                                    "WaveFront_win.dll" if sys.platform.startswith("win") else "libWaveFront.so"))
    _WaveFront = cdll.WaveFront
    _WaveFront.restype = None

    _WaveFront_mp = cdll.WaveFront_mp
    _WaveFront_mp.restype = None

    # (int32* world_map, long height, long width, long* pt_lst, long pt_num, int32* out)
    def wavefront(world_map, pt_lst, multiprocess=True):
        """
        :param world_map: (numpy.int32 h*w)
        :param pt_lst: (numpy 2*length)
        :param multiprocess:
        :return: (id_num, height, height)
        """
        height, width = world_map.shape
        pt_num = len(pt_lst)
        world_map_type = ctypes.POINTER(ctypes.c_int32)
        pt_type = ctypes.POINTER(ctypes.c_long)
        out_type = ctypes.c_int32 * (pt_num * width * height)
        argtype = [world_map_type, ctypes.c_long, ctypes.c_long, pt_type, ctypes.c_long, out_type]
        out = np.ones([pt_num * height * width], dtype=np.int32) * 99999
        out = out_type.from_buffer(out.data)
        world_map_in = world_map.reshape(-1).astype(np.int32).ctypes.data_as(world_map_type)
        pt_lst_in = pt_lst.reshape(-1).astype(np.long).ctypes.data_as(pt_type)
        # out = out_type()
        if multiprocess:
            function = _WaveFront_mp
        else:
            function = _WaveFront
        function.argtypes = argtype
        t1 = perf_counter()
        function(world_map_in, height, width, pt_lst_in, pt_num, out)
        t2 = perf_counter()
        print(t2 - t1)
        out = np.array(out).reshape([pt_num, height, width])
        return out

if __name__ == "__main__":
    import cv2

    world_map = np.array([0, 0, 0, 0, 0, 0, 0, 0,
                          0, 0, 0, 0, 0, 0, 0, 0,
                          0, 0, 0, 0, 0, 0, 0, 0,
                          1, 1, 1, 1, 1, 1, 0, 0,
                          0, 0, 0, 0, 0, 0, 0, 0,
                          0, 0, 0, 0, 0, 0, 0, 0,
                          0, 0, 0, 0, 0, 0, 0, 0,
                          0, 0, 0, 0, 0, 0, 0, 0,
                          ]).reshape([8, 8])
    # world_map = cv2.imread("/home/xint/mnt/wrinkle_registration_tool/exp/tr.tif", cv2.IMREAD_GRAYSCALE)
    pt_lst = np.array([[1, 1]])
    a = wavefront(world_map, pt_lst)
    # topographic_map = np.zeros_like(world_map)
    # topographic_map[a[0] % 100 == 0] = 255
    # cv2.imwrite("/home/xint/mnt/wrinkle_registration_tool/exp/topographic_map.jpg", topographic_map)
    print("finished")
