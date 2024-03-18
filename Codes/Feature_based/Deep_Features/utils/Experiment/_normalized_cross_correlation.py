import numpy as np
"""
function NCC=compute_NCC(img1,img2)
    meanA=sum(sum(img1))/(size(img1,1)*size(img1,2));
    meanB=sum(sum(img2))/(size(img2,1)*size(img2,2));
    Nup=sum(sum((img1-meanA).*(img2-meanB)));
    NA2=sum(sum((img1-meanA).*(img1-meanA)));
    NB2=sum(sum((img2-meanB).*(img2-meanB)));
    Ndown=sqrt(NA2*NB2);
    NCC=Nup/Ndown;
end
"""


def ncc(img1, img2):
    ncc_num = np.mean(np.multiply((img1 - np.mean(img1)), (img2 - np.mean(img2)))) / (np.std(img1) * np.std(img2))
    return ncc_num
