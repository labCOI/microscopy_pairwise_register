import argparse

parser = argparse.ArgumentParser(description='Label section waiting to align.')
parser.add_argument('--input', type=str, help='Input file.',
                    default=r'C:\Users\Xin\OneDrive\资料\paper\褶皱图像配准\论文稿件\els-cas-templates\pic\app_result\ea_0_.jpg')
parser.add_argument('--output', type=str, help='Output file.',
                    default=r'./ea_0_.png')
parser.add_argument('--label', type=str, help='Label file.',
                    default=r'G:\XT\PyCharmProjects\wrinkle_registration_tool\experiment\app_exp\lb_ea_rst.tif')
parser.add_argument('--layers', type=int, help='layer num.', default=31)
args = parser.parse_args()

if __name__ == '__main__':
    import os
    import numpy as np
    import skimage.io as io
    import cv2

    input_file = args.input
    output_file = args.output
    label_file = args.label
    layers = args.layers

    if not os.path.exists(input_file):
        raise FileNotFoundError('Input file not found!')
    if not os.path.exists(label_file):
        raise FileNotFoundError('Label file not found!')

    input_img = io.imread(input_file)
    label_img = io.imread(label_file)[layers, :, :]
    # resize label to input size using nearest interpolation
    label_img = cv2.resize(label_img, (input_img.shape[1], input_img.shape[0]), interpolation=cv2.INTER_NEAREST)
    # find no-zero points in label
    pts = np.array(np.where(label_img > 0)).T
    val = label_img[pts[:, 0], pts[:, 1]]
    input_img[label_img == 1, :] = [255, 0, 0]
    input_img[label_img == 2, :] = [0, 255, 0]
    io.imsave(output_file, input_img)