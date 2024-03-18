import cv2
import torch
import numpy as np
import matplotlib.pyplot as plt
from EA.WrinkleReg import WrinkleReg
from EA.superpoint_wrinkle_reg import SuperPointReg as SuperPoint
import torch.nn.functional as F
# from lightglue import LightGlue, SuperPoint, match_pair
# from lightglue.utils import *

from scipy.io import savemat
import time

from skimage.measure import ransac
from image_utils import *

import os
from os import listdir
 
def compute_transform_superlight(I1,I2, index_matrix):
    extractor = SuperPoint(max_num_keypoints=2048, detection_threshold=0.001).eval().cuda()
    matcher = LightGlue(features='superpoint').eval().cuda()
    I1t = numpy_image_to_torch(I1).cuda()
    I2t = numpy_image_to_torch(I2).cuda()
    t1 = time.time()
    status = 0
    e1 = extractor.extract(I1t)
    e2 = extractor.extract(I2t)
    matches = matcher({"image0": e1, "image1": e2})
    e1, e2, matches = [
        rbd(x) for x in [e1, e2, matches]
    ]
    pointsPrevious = e1['keypoints'].cpu().numpy()
    points = e2['keypoints'].cpu().numpy()
    pointsNumb = len(points)
    pointsPreviousNumb = len(pointsPrevious)
    featuresPrevious = e1['descriptors'].cpu().numpy()
    features = e2['descriptors'].cpu().numpy()

    matchedPointsPrev = e1['keypoints'][matches["matches"][..., 0]].cpu().numpy()
    matchedPoints = e2['keypoints'][matches["matches"][..., 1]].cpu().numpy()
    
    matchedNumb = len(matchedPoints)
    if matchedNumb>4:
        model_robust, inlierIndex = ransac((matchedPointsPrev, matchedPoints), EuclideanTransform, min_samples=2,
                               residual_threshold=1, max_trials=2000)
        if model_robust:
            inlierPoints = matchedPoints[np.ravel(inlierIndex)==1]
            inlierPointsPrev = matchedPointsPrev[np.ravel(inlierIndex)==1]
            index_ImMatch = index_matrix
            inlierNumb = np.sum(inlierIndex)
            X = model_robust.translation[0]
            Y = model_robust.translation[1]
            e = np.power((inlierPoints - inlierPointsPrev)-np.array([X,Y]),2)
            error = np.sqrt(np.sum(e)/np.sum(inlierIndex))
        else:
            print("transfrom not found")
            inlierNumb = np.nan
            status = 1
            index_ImMatch = np.nan
            X = np.nan
            Y = np.nan
            error = np.nan
    else:
        print("minimum points not found")
        status = 1
        inlierNumb = np.nan
        index_ImMatch = np.nan
        X = np.nan
        Y = np.nan
        error = np.nan
    t2 = time.time()
    out = np.array([Y, X, index_ImMatch, matchedNumb, pointsPreviousNumb, pointsNumb, inlierNumb, status, error, t2-t1])
    return out, pointsPrevious, featuresPrevious, points, features

def compute_transform_superBF(I1,I2, index_matrix):
    extractor = SuperPoint(max_num_keypoints=2048).eval().cuda()
    I1t = numpy_image_to_torch(I1).cuda()
    I2t = numpy_image_to_torch(I2).cuda()
    t1 = time.time()
    status = 0
    e1 = extractor.extract(I1t)
    e2 = extractor.extract(I2t)

    pointsPrevious = e1['keypoints'][0].cpu().numpy()
    points = e2['keypoints'][0].cpu().numpy()

    featuresPrevious = e1['descriptors'][0].cpu().numpy()
    features = e2['descriptors'][0].cpu().numpy()

    pointsNumb = len(points)
    pointsPreviousNumb = len(pointsPrevious)
    bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)
    matches = bf.match(featuresPrevious, features)

    matchedPointsPrev = np.float32([ pointsPrevious[m.queryIdx] for m in matches ]).reshape(-1,2)
    matchedPoints = np.float32([ points[m.trainIdx] for m in matches ]).reshape(-1,2)
    matchedNumb = len(matches)
    if matchedNumb>4:
        model_robust, inlierIndex = ransac((matchedPointsPrev, matchedPoints), EuclideanTransform, min_samples=2,
                               residual_threshold=1, max_trials=2000)
        if model_robust:
            inlierPoints = matchedPoints[np.ravel(inlierIndex)==1]
            inlierPointsPrev = matchedPointsPrev[np.ravel(inlierIndex)==1]
            index_ImMatch = index_matrix
            inlierNumb = np.sum(inlierIndex)
            X = model_robust.translation[0]
            Y = model_robust.translation[1]
            e = np.power((inlierPoints - inlierPointsPrev)-np.array([X,Y]),2)
            error = np.sqrt(np.sum(e)/np.sum(inlierIndex))
        else:
            print("transfrom not found")
            inlierNumb = np.nan
            status = 1
            index_ImMatch = np.nan
            X = np.nan
            Y = np.nan
            error = np.nan
    else:
        print("minimum points not found")
        status = 1
        inlierNumb = np.nan
        index_ImMatch = np.nan
        X = np.nan
        Y = np.nan
        error = np.nan
    t2 = time.time()
    out = np.array([Y, X, index_ImMatch, matchedNumb, pointsPreviousNumb, pointsNumb, inlierNumb, status, error, t2-t1])
    return out, pointsPrevious, featuresPrevious, points, features

def compute_transform(I1, I2, index_matrix):
    torch.random.manual_seed(0)
    np.random.seed(0)
    det = SuperPoint(thresh=0.001, patch_sz=512, batch_sz=8, overlap=0.8)
    t1 = time.time()
    status = 0
    [pointsPrevious, featuresPrevious] = det.detectAndComputeMean(I1)
    [points, features] = det.detectAndComputeMean(I2)

    pointsNumb = len(points)
    pointsPreviousNumb = len(pointsPrevious)
    bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)
    matches = bf.match(featuresPrevious, features)

    matchedPointsPrev = np.float32([ pointsPrevious[m.queryIdx].pt for m in matches ]).reshape(-1,2)
    matchedPoints = np.float32([ points[m.trainIdx].pt for m in matches ]).reshape(-1,2)
    matchedNumb = len(matches)
    if matchedNumb>4:
        model_robust, inlierIndex = ransac((matchedPointsPrev, matchedPoints), EuclideanTransform, min_samples=2,
                               residual_threshold=1, max_trials=2000)
        if model_robust:
            inlierPoints = matchedPoints[np.ravel(inlierIndex)==1]
            inlierPointsPrev = matchedPointsPrev[np.ravel(inlierIndex)==1]
            index_ImMatch = index_matrix
            inlierNumb = np.sum(inlierIndex)
            X = model_robust.translation[0]
            Y = model_robust.translation[1]
            e = np.power((inlierPoints - inlierPointsPrev)-np.array([X,Y]),2)
            error = np.sqrt(np.sum(e)/np.sum(inlierIndex))
        else:
            print("transfrom not found")
            inlierNumb = np.nan
            status = 1
            index_ImMatch = np.nan
            X = np.nan
            Y = np.nan
            error = np.nan
    else:
        print("minimum points not found")
        status = 1
        inlierNumb = np.nan
        index_ImMatch = np.nan
        X = np.nan
        Y = np.nan
        error = np.nan
    t2 = time.time()
    out = np.array([Y, X, index_ImMatch, matchedNumb, pointsPreviousNumb, pointsNumb, inlierNumb, status, error, t2-t1])
    return out, pointsPrevious, featuresPrevious, points, features

def compute_transform_west(I1, I2, index_matrix, X_pixel, OvX, M, N):
    mask = np.zeros_like(I1)
    mask[0:M, 0:X_pixel] = 255
    mask2 = np.zeros_like(I2)
    mask2[0:M, int(N*(1-OvX)):N] = 255

    torch.random.manual_seed(0)
    np.random.seed(0)
    det = SuperPoint(thresh=0.001, patch_sz=512, batch_sz=8, overlap=0.8) 
    t1 = time.time()
    status = 0
    [pointsPrevious, featuresPrevious] = det.detectAndComputeMean(I1, mask=mask)
    [points, features] = det.detectAndComputeMean(I2, mask=mask2)
    
    pointsNumb = len(points)
    pointsPreviousNumb = len(pointsPrevious)
    bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)
    matches = bf.match(featuresPrevious, features)

    matchedPointsPrev = np.float32([ pointsPrevious[m.queryIdx].pt for m in matches ]).reshape(-1,2)
    matchedPoints = np.float32([ points[m.trainIdx].pt for m in matches ]).reshape(-1,2)
    matchedNumb = len(matches)
    if matchedNumb>4:
        model_robust, inlierIndex = ransac((matchedPointsPrev, matchedPoints), EuclideanTransform, min_samples=2,
                               residual_threshold=1, max_trials=2000)        
        if model_robust:
            inlierPoints = matchedPoints[np.ravel(inlierIndex)==1]
            inlierPointsPrev = matchedPointsPrev[np.ravel(inlierIndex)==1]
            index_ImMatch = index_matrix
            inlierNumb = np.sum(inlierIndex)
            X = model_robust.translation[0]
            Y = model_robust.translation[1]
            e = np.power((inlierPoints - inlierPointsPrev)-np.array([X,Y]),2)
            error = np.sqrt(np.sum(e)/np.sum(inlierIndex))
        else:
            print("transfrom not found")
            pointsPrevious = np.nan
            pointsNumb = np.nan
            inlierNumb = np.nan
            status = 1
            index_ImMatch = np.nan
            X = np.nan
            Y = np.nan
            error = np.nan
            matchedNumb = np.nan
    else:
        print("minimum points not found")
        status = 1
        pointsPrevious = np.nan
        pointsNumb = np.nan
        inlierNumb = np.nan

        index_ImMatch = np.nan
        X = np.nan
        Y = np.nan
        error = np.nan
        matchedNumb = np.nan
    t2 = time.time()
    out = np.array([Y, X, index_ImMatch, matchedNumb, pointsPreviousNumb, pointsNumb, inlierNumb, status, error, t2-t1])
    return out, pointsPrevious, featuresPrevious, points, features, M

def compute_transform_north(I1, I2, index_matrix, Y_pixel, OvY, M, N):
    mask = np.zeros_like(I1)
    mask[0:Y_pixel,0:N] = 255
    mask2 = np.zeros_like(I1)
    mask2[int(M*(1-OvY)):M,0:N] = 255

    torch.random.manual_seed(0)
    np.random.seed(0)
    det = SuperPoint(thresh=0.001, patch_sz=512, batch_sz=8, overlap=0.8)

    t1 = time.time()
    status = 0
    [pointsPrevious, featuresPrevious] = det.detectAndComputeMean(I1, mask=mask)
    [points, features] = det.detectAndComputeMean(I2, mask=mask2)

    pointsNumb = len(points)
    pointsPreviousNumb = len(pointsPrevious)
    bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)
    matches = bf.match(featuresPrevious, features)

    matchedPointsPrev = np.float32([ pointsPrevious[m.queryIdx].pt for m in matches ]).reshape(-1,2)
    matchedPoints = np.float32([ points[m.trainIdx].pt for m in matches ]).reshape(-1,2)
    matchedNumb = len(matches)
    # temp_image = cv2.drawMatches(I1,pointsPrevious,I2,points,matches,None,flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    # plt.figure()
    # plt.imshow(temp_image)
    # plt.title(index_matrix)
    # plt.show()
    if matchedNumb>4:
        model_robust, inlierIndex = ransac((matchedPointsPrev, matchedPoints), EuclideanTransform, min_samples=2,
                               residual_threshold=1, max_trials=2000)
        if model_robust:
            inlierPoints = matchedPoints[np.ravel(inlierIndex)==1]
            inlierPointsPrev = matchedPointsPrev[np.ravel(inlierIndex)==1]
            index_ImMatch = index_matrix
            inlierNumb = np.sum(inlierIndex)
            X = model_robust.translation[0]
            Y = model_robust.translation[1]
            e = np.power((inlierPoints - inlierPointsPrev)-np.array([X,Y]),2)
            error = np.sqrt(np.sum(e)/np.sum(inlierIndex))
        else:
            print("transfrom not found")
            pointsPrevious = np.nan
            pointsNumb = np.nan
            inlierNumb = np.nan
            status = 1
            index_ImMatch = np.nan
            X = np.nan
            Y = np.nan
            error = np.nan
            matchedNumb = np.nan
    else:
        print("minimum points not found")
        status = 1
        pointsPrevious = np.nan
        pointsNumb = np.nan
        inlierNumb = np.nan

        index_ImMatch = np.nan
        X = np.nan
        Y = np.nan
        error = np.nan
        matchedNumb = np.nan
    t2 = time.time()
    out = np.array([Y, X, index_ImMatch, matchedNumb, pointsPreviousNumb, pointsNumb, inlierNumb, status, error, t2-t1])
    return out, pointsPrevious, featuresPrevious, points, features, M



