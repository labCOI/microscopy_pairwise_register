# A comparative analysis of pairwise image stitching techniques for microscopy images

## Introduction
Stitching of microscopic images is a technique used to combine multiple overlapping images (tiles) from biological samples with a limited field of view and high resolution to create a whole slide image. Image stitching involves two main steps: pairwise registration and global alignment. Most of the computational load and the accuracy of the stitching algorithm depend on the pairwise registration method. Therefore, choosing an efficient, accurate, robust, and fast pairwise registration method is crucial in the whole slide imaging technique. This paper presents a detailed comparative analysis of different pairwise registration techniques in terms of execution time and quality.

## Objective
The objective of this study is to compare the performance of different feature-based and region-based pairwise registration methods on microscopic images from various modalities such as bright-field, phase-contrast, and fluorescence. Feature-based methods include Harris, Shi-Thomas, FAST, ORB, BRISK, SURF, SIFT, KAZE, MSER, and deep learning-based SuperPoint features. Additionally, region-based methods based on the normalized cross-correlation (NCC) and combination of phase correlation and NCC are investigated in terms of execution time and accuracy.

## Methods
This study compares region-based pairwise registration methods (NCC and Phase-NCC) with feature-based methods (Harris, Shi-Thomasi, FAST, ORB, BRISK, SURF, SIFT, KAZE, MSER, and SuperPoint features). The investigation is conducted on experimental microscopy images to assess accuracy, processing speed, and robustness to uneven illumination.

## Results
Feature-based methods outperformed region-based methods in terms of accuracy and processing speed. Additionally, feature-based methods were highly robust to uneven illumination of tiles. Among feature-based methods, SURF features were identified as the most effective, surpassing all other techniques on different image modalities including bright-field, phase-contrast, and fluorescence.

## Conclusion
This study provides valuable insights into the strengths and weaknesses of different registration methods for microscopic image stitching. Researchers in the field of computational pathology and biology can utilize these findings to select the most suitable pairwise registration method for creating whole slide images. 

## Citation
If you find our paper and code useful, please kindly cite our paper as follows:

```bibtex
@article{mohammadi2024,
  author = {Fatemeh Sadat Mohammadi and Seyyed Erfan Mohammadi and Parsa Mojarad Ali and Seyed Mohammad Ali Mirkarimi and Hasti Shabani},
  title = {A Comparative Analysis of Pairwise Image Stitching Techniques for Microscopy Images},
  journal = {Scientific Reports},
  volume = {14},
  pages = {9215},
  year = {2024},
  doi = {10.1038/s41598-024-59626-y}
}

