import torch
import torch.nn.functional as F
import cv2
import numpy as np
from utils.convert import *


# input must be 4-dim
class Flow:
    def __init__(self, field=None, sz=None, torch_like=True):
        """
        光流场

        :param field: 光流场，可以是绝对的，相对的，以及仿射变换的矩阵，[B,2,N,N] or[B,N,N,2]
        :param sz: 如果输入仿射变换矩阵，就要确定光流场的大小
        :param torch_like: 是否为pytorch一样的绝对坐标代表的形变场
        """
        theta = torch.tensor([[1, 0, 0], [0, 1, 0]],device=field.device).float().unsqueeze(0)
        if sz is None:
            """
            光流场
            """
            if field.shape[-1] != 2:
                field = field.permute([0, 2, 3, 1])
            sz = field.shape[:3]
            self.base_field = F.affine_grid(theta, [1, 1, sz[-2], sz[-1]], align_corners=False)
            if torch_like:
                self.field = field
            else:
                self.field = field + self.base_field
        else:
            """
            仿射矩阵
            """
            self.base_field = F.affine_grid(theta, [1, 1, sz[-2], sz[-1]], align_corners=False)
            self.field = F.affine_grid(field, [1, 1, sz[-2], sz[-1]], align_corners=False)

    # return a numpy
    def draw_flow(self, alpha=0):
        assert self.field is not None, "please create a field"
        field = self.field.detach() - self.base_field
        pi = 3.141592653
        deltay = field[:, :, :, 1:]
        deltax = field[:, :, :, :1]
        if alpha == 0.:
            alpha = 1. / ((deltax ** 2 + deltay ** 2) ** 0.5 / 2).cpu().max()
        h = (torch.atan2(deltay, deltax) / (2 * pi) + (deltay < 0) * 1.).cpu()
        s = alpha * ((deltax ** 2 + deltay ** 2) ** 0.5 / 2).cpu()
        v = torch.ones(h.shape)
        color_field = torch.cat([h, s, v], dim=-1)
        l = []
        for f in color_field.numpy():
            l.append(cv2.cvtColor((f*255).astype(np.uint8), cv2.COLOR_HSV2BGR_FULL))
        color_field = np.array(l)
        return color_field

    def sample(self, src, mode="'bilinear'"):
        """
        :param src: numpy
        :return: numpy
        """
        assert self.field is not None, "please create a field"
        grid = self.field
        src = Any2Torch(src).to(grid.device)
        warped = F.grid_sample(src, grid,align_corners=False,mode=mode)
        return np.array(Any2PIL(warped))

# __all__ = {"Flow"}
# if __name__ == "__main__":
#     theta = torch.tensor([[1, 0, 0], [0, 1, 0]]).float().unsqueeze(0)
#     field = F.affine_grid(theta, [1, 1, 800, 800])
#     flow = Flow(field * 2)
#     flow_color = flow.draw_flow()
#     cv2.imwrite("/home/xint/mnt/inpaint/temp_rst/flow.tif", flow_color[0])
#     print("finished")
