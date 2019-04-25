import numpy as np
import cv2
import os
import json

def readLandmarks(fname):
    with open(fname) as f:
        line = f.readline()
        nItems = int(line)
        res = np.zeros((nItems, 2), dtype=np.float32)
        for i, s in enumerate(f):
            res[i] = map(float, (x for x in s.split(' ') if len(x) > 0))
    return res

def drawLandmarks(img, landmarks, scale = 1.0, withNumbers = False):
    for i, x in enumerate(landmarks):
        center = (int(x[0] * scale), int(img.shape[0] - scale*x[1]))
        cv2.circle(
                img, center, 2, (0, 255, 0), 1
                )
        if withNumbers:
            font = cv2.FONT_HERSHEY_SIMPLEX
            cv2.putText(img, str(i), center, font, 0.7, (0, 0, 255), 1)

def fillPolygon(img, landmarks, scale, shift = (0, 0), color = (0, 255, 255), closedCurve=True):
    pts = []
    for i, x in enumerate(landmarks):
        center = (int(x[0] * scale + shift[0]), int(shift[1] + img.shape[0] - scale*x[1]))
        pts.append(center)
    #pts = np.array(pts).reshape((-1, 1, 2))
    pts = np.array(pts).reshape((1, -1, 2))

    #for p in pts:
    #    cv2.circle(img, (p[0, 0], p[0, 1]), 3, color, 4)
    #cv2.circle(img, (pts[4, 0, 0], pts[4, 0, 1]), 3, color, 4)
    cv2.polylines(img, pts, closedCurve, color, 2)
    #cv2.fillPoly(img,[pts],color)

def landmarkBorder(landmarks, centeredScale = 1.0, rectify = False):
    ptLeftUp = np.array([1e4, 1e4])
    ptRightBottom = np.array([-1.0, -1.0])
    for p in landmarks:
        ptLeftUp[0] = ptLeftUp[0] if p[0] > ptLeftUp[0] else p[0]
        ptLeftUp[1] = ptLeftUp[1] if p[1] > ptLeftUp[1] else p[1]
        ptRightBottom[0] = ptRightBottom[0] if p[0] < ptRightBottom[0] else p[0]
        ptRightBottom[1] = ptRightBottom[1] if p[1] < ptRightBottom[1] else p[1]
    
    middle = (ptRightBottom - ptLeftUp) / 2.0 + ptLeftUp
    ptLeftUp = middle + centeredScale * (ptLeftUp - middle)
    ptRightBottom = middle + centeredScale * (ptRightBottom - middle)
    if rectify:
        ptDiff = ptRightBottom - ptLeftUp
        biggestDim = max(ptDiff)
        ptLeftUp = middle - biggestDim/2.0
        ptRightBottom = middle + biggestDim/2.0
    return ptLeftUp, ptRightBottom

def cutImage(img, p1, p2):
    p2[1] = img.shape[0] - p2[1]
    p1[1] = img.shape[0] - p1[1]
    if p1[1] > p2[1]:
        p1[1], p2[1] = p2[1], p1[1]

    p1 = map(int, p1)
    p2 = map(int, p2)
    return img[p1[1] : p2[1], p1[0] : p2[0] , :]


leftEyeIndices = [27, 66, 28, 69, 29, 68, 30, 67]
leftHiEyeLidIndices = [27, 66, 28, 69, 29]
leftLoEyeLidIndices = [27, 67, 30, 68, 29]

mouthAllIndices = [
    46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57
]

faceOutterIndices = [
    0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14,
    15, 16, 22, 21
]

DATA_ROOT = "/home/daiver/Downloads/FaceWarehouse/"

#IMGS_ROOT = DATA_ROOT + "FaceWarehouse_imgs/"
#LNDS_ROOT = DATA_ROOT + "FaceWarehouse_landmarks/"

nPersons = 150

samples = []


classColorCodes = [
    (255, 255, 255)
]

def createLeftEyelidsSegmentationMask(img, lnd):
    #eyeLnd = lnd[faceOutterIndices, :]
    #eyeLnd = lnd[mouthAllIndices, :]
    eyeLnd   = lnd[leftEyeIndices, :]
    eyeHiLnd = lnd[leftHiEyeLidIndices, :]
    eyeLoLnd = lnd[leftLoEyeLidIndices, :]
    img = cv2.pyrUp(cv2.pyrUp(img))

    p1, p2 = landmarkBorder(eyeLnd, 1.5, True)

    img2 = cutImage(img, p1*4, p2*4)
    outFileName = "Tester_%s_%s_pose_%s" % (personInd, subdir, poseInd)
    imgOutName = "data/imgs/Img_%s.png" % outFileName
    cv2.imwrite(imgOutName, img2)

    label = np.zeros((img2.shape[0], img2.shape[1], 3), dtype=np.uint8)
    #fillPolygon(label, eyeLnd - p1, 4, (0, 0), (255, 255, 255), closedCurve=True)
    #label = cv2.dilate(label, None)


    label1 = np.copy(label)
    label2 = np.copy(label)
    fillPolygon(label1, eyeHiLnd - p1, 4, (0, 0), (255, 255, 255), closedCurve=False)
    fillPolygon(label2, eyeLoLnd - p1, 4, (0, 0), (255, 255, 255), closedCurve=False)
    label1 = cv2.dilate(label1, None)
    label2 = cv2.dilate(label2, None)
    label  = np.zeros_like(label1)
    #label  = 
    #label[:, :, 0] = label1[:, :, 0]
    #label[:, :, 1] = label2[:, :, 0]

    lblOutName  = "data/labels/Lbl_%s_label.png" % (outFileName)
    cv2.imwrite(lblOutName, label)
    lblOutName0 = "data/labels/Lbl_%s_label_%s.png" % (outFileName, 0)
    cv2.imwrite(lblOutName0, label1)
    lblOutName1 = "data/labels/Lbl_%s_label_%s.png" % (outFileName, 1)
    cv2.imwrite(lblOutName1, label2)

    paths2Label = [lblOutName0, lblOutName1]
    return imgOutName, paths2Label, lblOutName

def createLeftEyelidsSegmentationMaskLarge(img, lnd):
    #eyeLnd = lnd[faceOutterIndices, :]
    #eyeLnd = lnd[mouthAllIndices, :]
    eyeLnd   = lnd[leftEyeIndices, :]
    eyeHiLnd = lnd[leftHiEyeLidIndices, :]
    eyeLoLnd = lnd[leftLoEyeLidIndices, :]
    scaleCoeff = 1
    #img = cv2.pyrUp(cv2.pyrUp(img))

    p1, p2 = landmarkBorder(lnd, 1.5, True)

    img2 = cutImage(img, p1*scaleCoeff, p2*scaleCoeff)
    outFileName = "Tester_%s_%s_pose_%s" % (personInd, subdir, poseInd)
    imgOutName = "data/imgs/Img_%s.png" % outFileName
    cv2.imwrite(imgOutName, img2)

    label = np.zeros((img2.shape[0], img2.shape[1], 3), dtype=np.uint8)
    #fillPolygon(label, eyeLnd - p1, 4, (0, 0), (255, 255, 255), closedCurve=True)
    #label = cv2.dilate(label, None)


    label1 = np.copy(label)
    label2 = np.copy(label)
    fillPolygon(label1, eyeHiLnd - p1, scaleCoeff, (0, 0), (255, 255, 255), closedCurve=False)
    fillPolygon(label2, eyeLoLnd - p1, scaleCoeff, (0, 0), (255, 255, 255), closedCurve=False)
    #label1 = cv2.dilate(label1, None)
    #label2 = cv2.dilate(label2, None)

    #label1 = cv2.dilate(label1, None)
    #label2 = cv2.dilate(label2, None)

    #label1 = cv2.dilate(label1, None)
    #label2 = cv2.dilate(label2, None)

    #label1 = cv2.dilate(label1, None)
    #label2 = cv2.dilate(label2, None)

    label  = np.copy(img2).astype(np.float32)
    label[:, :, 0] += label1[:, :, 0] 
    label[:, :, 1] += label2[:, :, 0] 
    #label  = 
    #label[:, :, 0] = label1[:, :, 0]
    #label[:, :, 1] = label2[:, :, 0]

    lblOutName  = "data/labels/Lbl_%s_label.png" % (outFileName)
    cv2.imwrite(lblOutName, label)
    lblOutName0 = "data/labels/Lbl_%s_label_%s.png" % (outFileName, 0)
    cv2.imwrite(lblOutName0, label1)
    lblOutName1 = "data/labels/Lbl_%s_label_%s.png" % (outFileName, 1)
    cv2.imwrite(lblOutName1, label2)

    paths2Label = [lblOutName0, lblOutName1]
    return imgOutName, paths2Label, lblOutName


for personInd in xrange(1, nPersons + 1):
    personName = "Tester_%s" % (personInd)
    personDir  = os.path.join(DATA_ROOT, personName)
    subdirs = os.listdir(personDir)
    for subdir in subdirs:
        absSubdir = os.path.join(personDir, subdir)
        print absSubdir
        for poseInd in xrange(0, 24):
            imgName = os.path.join(absSubdir, "pose_%s.jpg"  % poseInd)
            lndName = os.path.join(absSubdir, "pose_%s.land" % poseInd)
            #print imgName
            img = cv2.imread(imgName)
            lnd = readLandmarks(lndName)

            #imgOutName, paths2Label, lblOutName = createLeftEyelidsSegmentationMask(img, lnd)
            imgOutName, paths2Label, lblOutName = createLeftEyelidsSegmentationMaskLarge(img, lnd)

            samples.append({
                'Tester' : personInd,
                'subdir' : subdir,
                'pose'   : poseInd,
                'path2Img' : imgOutName,
                'paths2Label' : paths2Label,
                'path2VisLabel' : lblOutName
            })

            #cv2.imshow('1', img2)
            #cv2.imshow('2', label)

            #img = cv2.imread(imgName)
            #img = cv2.pyrUp(cv2.pyrUp(img))
            #drawLandmarks(img, lnd, 4, True)
            #cv2.imwrite('landmarksfacewarehousedrawed.png', img)
            #exit(0)
            #fillPolygon(img, eyeLnd, 4)
            #cv2.imshow('', img)

            #cv2.waitKey()

dataDesc = {
    'samples' : samples,
    'classColorCodes' : classColorCodes
}

with open('data/data.txt', 'w') as outfile:
    json.dump(dataDesc, outfile)

print 'Finished! nSamples', len(samples)
