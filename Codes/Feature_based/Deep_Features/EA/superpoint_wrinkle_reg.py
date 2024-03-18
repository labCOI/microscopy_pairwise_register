import numpy as np
from utils.convert import lst2kpl
from utils.superpoint import SuperPoint, PATH, SuperPointFrontend


def heat_map2kp(self, heatmap):
    height = heatmap.shape[0]
    width = heatmap.shape[1]
    ys, xs = np.where(heatmap >= self.conf_thresh)  # Confidence threshold.
    if len(ys) == 0:
        return np.zeros((3, 0))
    pts = np.zeros((3, len(ys)))  # Populate point data sized 3xN.
    pts[0, :] = xs
    pts[1, :] = ys
    pts[2, :] = heatmap[ys, xs]
    pts, _ = self.nms_fast(pts, height, width, dist_thresh=self.nms_dist)  # Apply NMS.
    inds = np.argsort(pts[2, :])
    pts = pts[:, inds[::-1]]  # Sort by confidence.
    # Remove points along border.
    bord = self.border_remove
    toremoveW = np.logical_or(pts[0, :] < bord, pts[0, :] >= (width - bord))
    toremoveH = np.logical_or(pts[1, :] < bord, pts[1, :] >= (height - bord))
    toremove = np.logical_or(toremoveW, toremoveH)
    pts = pts[:, ~toremove]
    return pts


class SuperPointReg(SuperPoint):
    def detectAndComputeMean(self, img, scale=1, mask=None, grid_n=10, max_num=1000):
        heat_map, des_map = self.get_heat_map(img, mask)
        block_max_num = max_num // (grid_n ** 2)
        self.sp_net.border_remove = 0
        height, width = heat_map.shape
        step_h = height // grid_n
        step_w = width // grid_n
        pts_lst = []
        for j in range(grid_n):
            for i in range(grid_n):
                block_map = heat_map[j * step_h:(j + 1) * step_h, i * step_w:(i + 1) * step_w]
                pts = self.sp_net.heat_map2kp(block_map)[:, :block_max_num]
                pts[0] = pts[0] + i * step_w
                pts[1] = pts[1] + j * step_h
                pts_lst.append(pts)
        pts = np.concatenate(pts_lst,axis=1).astype(np.float)
        # Remove points along border.
        bord = 4
        toremoveW = np.logical_or(pts[0, :] < bord, pts[0, :] >= (width - bord))
        toremoveH = np.logical_or(pts[1, :] < bord, pts[1, :] >= (height - bord))
        toremove = np.logical_or(toremoveW, toremoveH)
        kp = pts[:, ~toremove]
        desc = self.get_desc(kp, des_map, img)
        kp = kp*scale
        kpl = lst2kpl(kp[:2].transpose())
        return kpl, desc
