import os
import cv2
import numpy as np
import random
import math

defaultP1 = np.array([-0.5, -1./3], dtype=np.float32)
defaultP2 = np.array([+0.5, -1./3], dtype=np.float32)
defaultP3 = np.array([ 0, +2./3], dtype=np.float32)

def loadData(dirName):
    leftEyeIndex  = 16
    rightEyeIndex = 17
    mouthIndex    = 24
    fileNames = os.listdir(dirName)
    imagesNames = filter(lambda x: x[-4:] == '.png', fileNames)
    dataNames   = filter(lambda x: x[-4:] == '.txt', fileNames)
    imagesNames.sort()
    dataNames.sort()
    assert(len(dataNames) == len(imagesNames))
    images = []
    bboxes = []
    leftEyeValues  = []
    rightEyeValues = []
    mouthValues    = []
    names          = []
    for imgName, dataName in zip(imagesNames, dataNames):
        lines = open(os.path.join(dirName, dataName)).readlines()

        bbox = [int(float(x)) for x in lines[0].split()]
        leftEyeCoords  = map(float, lines[1 + leftEyeIndex ].split())
        rightEyeCoords = map(float, lines[1 + rightEyeIndex].split())
        mouthCoords    = map(float, lines[1 + mouthIndex   ].split())
        img = cv2.imread(os.path.join(dirName, imgName), 0)

        images.append(img)
        bboxes.append(bbox)
        leftEyeValues.append(leftEyeCoords)
        rightEyeValues.append(rightEyeCoords)
        mouthValues.append(mouthCoords)
        names.append(imgName)

    return images, bboxes, leftEyeValues, rightEyeValues, mouthValues, names

def makeTrainDataFromRawData(images, bboxes, values, countOfCoords, samplesForOneImage):
    resData   = []
    resValues = [[] for x in xrange(countOfCoords)]
    for img, bbox, val in zip(images, bboxes, values):
        for i in xrange(samplesForOneImage):
            sample = [float(bbox[0] + random.randint(0, int(bbox[2]/2))), 
                      float(bbox[1] + random.randint(0, int(bbox[3]/2)))]

            resData.append((img, sample))
            for j in xrange(countOfCoords):
                resValues[j].append(val[j])

#            cv2.circle(img, (sample[0], sample[1]), 5, 255)
            #cv2.imshow('', img)
            #cv2.waitKey()

    return resData, resValues

def makeTrainDataFromRawDataTriangles(
        images, bboxes, values1, values2, values3, samplesForOneImage):
    resData   = []
    countOfCoords = 5
    resValues = [[] for x in xrange(countOfCoords)]
    for index, (img, bbox, p1, p2, p3) in enumerate(
            zip(images, bboxes, values1, values2, values3)):
        pose = computePoseFromTriangle(p1, p2, p3)
        #print pose
        for i in xrange(samplesForOneImage):
            sample = randomSampleFromBBox(bbox)

            resData.append((img, sample, index))
            for j in xrange(countOfCoords):
                resValues[j].append(pose[j])

#            dp1 = apply5DTransformation(sample, defaultP1)
            #dp2 = apply5DTransformation(sample, defaultP2)
            #dp3 = apply5DTransformation(sample, defaultP3)

            #tp1 = apply5DTransformation(pose, defaultP1)
            #tp2 = apply5DTransformation(pose, defaultP2)
            #tp3 = apply5DTransformation(pose, defaultP3)


            #imgT = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

            #cv2.rectangle(imgT, (int(bbox[0]), int(bbox[1])),
                                 #(int(bbox[0] + bbox[2]), int(bbox[1] + bbox[3])),
                                 #(0, 0, 255))

            ##print sample

            #cv2.circle(imgT, (int(dp1[0]), int(dp1[1])), 5, (0, 255, 0))
            #cv2.circle(imgT, (int(dp2[0]), int(dp2[1])), 5, (0, 255, 0))
            #cv2.circle(imgT, (int(dp3[0]), int(dp3[1])), 5, (0, 255, 0))

            #cv2.circle(imgT, (int(tp1[0]), int(tp1[1])), 5, (255,0, 0))
            #cv2.circle(imgT, (int(tp2[0]), int(tp2[1])), 5, (255,0, 0))
            #cv2.circle(imgT, (int(tp3[0]), int(tp3[1])), 5, (255,0, 0))


            #cv2.imshow('', imgT)
            #cv2.waitKey()
            #cv2.destroyAllWindows()

    #print 'First sample', resData[0][1]

    return resData, resValues

def computePoseFromTriangle(p1, p2, p3):
    translate = np.average([p1, p2, p3], axis=0)
    p1 = p1 - translate
    p2 = p2 - translate
    p3 = p3 - translate

    AX = np.array([
        [defaultP1[0], -defaultP1[1]],
        [defaultP2[0], -defaultP2[1]],
        [defaultP3[0], -defaultP3[1]]], dtype=np.float32)
    BX = np.array([p1[0], p2[0], p3[0]], dtype=np.float32)

    AY = np.array([
        [defaultP1[0], defaultP1[1]],
        [defaultP2[0], defaultP2[1]],
        [defaultP3[0], defaultP3[1]]], dtype=np.float32)
    BY = np.array([p1[1], p2[1], p3[1]], dtype=np.float32)

    resX = np.linalg.lstsq(AX, BX)[0]
    resY = np.linalg.lstsq(AY, BY)[0]

    SxC, SyS = resX[0], resX[1]
    SxS, SyC = resY[0], resY[1]
    Sx = np.sqrt(SxS**2 + SxC**2)
    Sy = np.sqrt(SyS**2 + SyC**2)

    c = SxC / Sx
    s = SxS / Sx

    angle = math.atan2(s, c)
    return np.array([translate[0], translate[1], angle, Sx, Sy], dtype=np.float32)

def apply5DTransformation(pose, point):
    a = np.zeros((3, 3))
    a[2, 2] = 1
    sin = np.sin(pose[2])
    cos = np.cos(pose[2])
    a[0, 0] =  pose[3] * cos
    a[1, 0] =  pose[3] * sin
    a[0, 1] = -pose[4] * sin
    a[1, 1] =  pose[4] * cos
    a[0, 2] =  pose[0]
    a[1, 2] =  pose[1]

    return np.dot(a, np.array([point[0], point[1], 1.0]))[:2]

def randomSampleFromBBox(bbox):
    pose = np.array([
        bbox[0] + bbox[2]/2.0,
        bbox[1] + bbox[3]/2.0,
        0.0,
        bbox[2]/2.0,
        bbox[3]/2.0], dtype=np.float32)
    sample = pose
    sample[0] += random.uniform(-bbox[2]/4.0, bbox[2]/4.0)
    sample[1] += random.uniform(-bbox[3]/4.0, bbox[3]/4.0)
    sample[2] += random.uniform(-np.pi/5, np.pi/5)
    sample[3] *= random.uniform(0.9, 1.1)
    sample[4] *= random.uniform(0.9, 1.1)

    return sample

if __name__ == '__main__':
    images, bboxes, values1, values2, values3, names = loadData('/home/daiver/Downloads/COFW/dump/')
    makeTrainDataFromRawDataTriangles(images, bboxes, values1, values2, values3, 3)
    exit()
    print 'Start'
    for img, p1, p2, p3 in zip(images, values1, values2, values3):
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        p1 = np.array(p1, dtype=np.float32)
        p2 = np.array(p2, dtype=np.float32)
        p3 = np.array(p3, dtype=np.float32)
        pose = computePoseFromTriangle(p1, p2, p3)
        dp1 = apply5DTransformation(pose, defaultP1)
        dp2 = apply5DTransformation(pose, defaultP2)
        dp3 = apply5DTransformation(pose, defaultP3)
        dp4 = (dp1 + dp2 + dp3)/3

        #img = np.zeros(img.shape, dtype=np.uint8)

        cv2.circle(img, (int(p1[0]), int(p1[1])), 5, (255, 0, 0))
        cv2.circle(img, (int(p2[0]), int(p2[1])), 5, (255, 0, 0))
        cv2.circle(img, (int(p3[0]), int(p3[1])), 5, (255, 0, 0))

        cv2.circle(img, (int(dp1[0]), int(dp1[1])), 5, (0, 255, 0))
        cv2.circle(img, (int(dp2[0]), int(dp2[1])), 5, (0, 255, 0))
        cv2.circle(img, (int(dp3[0]), int(dp3[1])), 5, (0, 255, 0))
        cv2.circle(img, (int(dp4[0]), int(dp4[1])), 5, (0, 255, 0))

        cv2.imshow('', img)
        cv2.waitKey()
        cv2.destroyAllWindows()

