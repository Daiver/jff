import math
import random
import numpy as np
import cv2
import os

def defAugSettings():
    return {
        'scaleMin'      :  0.833333333,
        'scaleMax'      :  1.2,
        'angleMin'      : -9.0,
        'angleMax'      :  9.0,
        'tXRelativeMin' : -0.2,
        'tXRelativeMax' :  0.2,
        'tYRelativeMin' : -0.2,
        'tYRelativeMax' :  0.2,
    }

def randomTransMatrix(settings, width, height):
    s = settings

    angle = random.uniform(s['angleMin'], s['angleMax']) 
    tX    = random.uniform(s['tXRelativeMin'], s['tXRelativeMax']) * width
    tY    = random.uniform(s['tYRelativeMin'], s['tYRelativeMax']) * height
    tXr   = random.uniform(s['tXRelativeMin'], s['tXRelativeMax']) * width  + width/2.0
    tYr   = random.uniform(s['tYRelativeMin'], s['tYRelativeMax']) * height + height/2.0
    scale = random.uniform(s['scaleMin'], s['scaleMax'])

    res = cv2.getRotationMatrix2D((tXr, tYr), angle, scale)
    res[:2, 2] = [tX, tY]

    return res

def augmentSegmentation(settings, dataList, includeOriginal=False, nAugTimes=1):
    nItems = dataList[0].shape[0]
    nData = len(dataList)
    res = []
    if includeOriginal:
        nAugTimes += 1
    for x in dataList:
        dtype = x.dtype
        shape = x.shape
        shape = (shape[0] * nAugTimes, ) + shape[1:]
        tmp = np.zeros(shape, dtype=dtype)
        res.append(tmp)
    if includeOriginal:
        for i, x in enumerate(dataList):
            shape = x.shape
            res[i][:shape[0]] = x

    st = 1 if includeOriginal else 0
    for k in xrange(st, nAugTimes):
        for i in xrange(nItems):
            h, w = dataList[0].shape[1], dataList[0].shape[2]
            mat = randomTransMatrix(settings, w, h)
            for j in xrange(nData):
                item = dataList[j][i]
                deformed = cv2.warpAffine(item, mat, (w, h)).reshape(item.shape)
                #print i, j, k, item.shape, deformed.shape
                res[j][i + nItems*k] = deformed
    return res


if __name__ == '__main__':
    imgName1 = 'data/imgs/Img_Tester_145_pose_9.png'
    lblName1 = 'data/labels/Lbl_Tester_145_pose_9.png'
    imgName2 = 'data/imgs/Img_Tester_45_pose_3.png'
    lblName2 = 'data/labels/Lbl_Tester_45_pose_3.png'
    img1 = cv2.imread(imgName1)
    lbl1 = cv2.imread(lblName1, 0).astype(np.float32) / 255.0
    img2 = cv2.imread(imgName2)
    lbl2 = cv2.imread(lblName2, 0).astype(np.float32) / 255.0

    targetSize = (128, 128)
    img1 = cv2.resize(img1, targetSize)
    img2 = cv2.resize(img2, targetSize)
    lbl1 = cv2.resize(lbl1, targetSize)
    lbl2 = cv2.resize(lbl2, targetSize)

    imgs = np.stack([img1, img2])
    lbls = np.stack([lbl1, lbl2])

    nAugTimes = 5
    print imgs.shape
    aImgs, aLbls = augmentSegmentation(
            defAugSettings(), [np.array(imgs), np.array(lbls)],
            True, nAugTimes)

    print len(aImgs), len(aLbls)

    imgs = np.concatenate([imgs] * nAugTimes)
    lbls = np.concatenate([lbls] * nAugTimes)
    print imgs.shape

    for img, lbl, aImg, aLbl in zip(imgs, lbls, aImgs, aLbls):
        cv2.imshow('img', img)
        cv2.imshow('lbl', lbl)
        cv2.imshow('aImg', aImg)
        cv2.imshow('aLbl', aLbl)
        cv2.waitKey()

