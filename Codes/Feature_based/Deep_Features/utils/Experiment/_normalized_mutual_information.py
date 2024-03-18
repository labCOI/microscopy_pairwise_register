# -*- coding: utf-8 -*-
# @Time    : 2021/5/11 16:59
# @Author  : XinTong
# @FileName: _normalized_mutual_information.py
# @Software: PyCharm
import numpy as np
import cv2

"""
function NMI=compute_NMI(img1,img2)
%计算归一化互信息
%         [img1,img2]=histq_sp(img1,img2);%直方图均衡化
        [Ma,Na] = size(img1);
        [Mb,Nb] = size(img2);
        m=min(Ma,Mb);
        n=min(Na,Nb); 
%         ET=entropy(img1);
%         ES=entropy(img2);%//模板熵
        histq=zeros(256,256);%//联合直方图，清空
        histqx=zeros(256,1);%//img1直方图，清空
        histqy=zeros(256,1);%//img2直方图，清空
        %//联合直方图
        for s=1:m
            for t=1:n
                x=img1(s,t)+1;y=img2(s,t)+1;%//灰度<—>坐标
                histq(x,y)=histq(x,y)+1;
                histqx(x)=histqx(x)+1;
                histqy(y)=histqy(y)+1;
            end
        end
        p=histq./sum(sum(histq));%//联合概率密度
        px=histqx./sum(histqx);
        py=histqy./sum(histqy);
        ES=sum(px.*log(px+eps)/log(2));
        ET=sum(py.*log(py+eps)/log(2));
        EST=sum(sum(p.*log(p+eps)/log(2)));
        I=ES+ET-EST;
        NMI=2*I/(ES+ET);%可以改成互信息或者归一化系数等等
end
"""

eps = 1e-16


def nmi(img1, img2):
    # img1 = cv2.equalizeHist(img1)
    # img2 = cv2.equalizeHist(img2)
    Ma, Na = img1.shape
    Mb, Nb = img2.shape
    m = min(Ma, Mb)
    n = min(Na, Nb)
    histq = np.zeros([256, 256])
    histqx = np.zeros(256)
    histqy = np.zeros(256)
    for s in range(m):
        for t in range(n):
            x = img1[s, t]
            y = img2[s, t]  # // 灰度 <— > 坐标
            histq[x, y] = histq[x, y] + 1
            histqx[x] = histqx[x] + 1
            histqy[y] = histqy[y] + 1
    p = histq / histq.sum()  # 联合概率密度
    px = histqx / histqx.sum()
    py = histqy / histqy.sum()
    ES = (np.multiply(px, np.log(px + eps)) / np.log(2)).sum()
    ET = (np.multiply(py, np.log(py + eps)) / np.log(2)).sum()
    EST = (np.multiply(p, np.log(p + eps)) / np.log(2)).sum()
    I = ES + ET - EST
    nmi_num = 2 * I / (ES + ET)  # 可以改成互信息或者归一化系数等等
    return nmi_num


if __name__ == "__main__":
    PATH = "/home/xint/mnt/wrinkle_registration_tool/"
    img1 = cv2.imread(PATH+"exp/reference.tif", cv2.IMREAD_GRAYSCALE)
    img2 = cv2.imread(PATH+"exp/moving.tif", cv2.IMREAD_GRAYSCALE)
    img3 = cv2.imread(PATH+"exp/moving_finished.tif", cv2.IMREAD_GRAYSCALE)
    nmi1 = nmi(img1,img2)
    nmi2 = nmi(img1,img3)
