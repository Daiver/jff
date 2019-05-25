import cv2
import numpy as np
import os
import sys

def readPhotos(path2photos, max_photos2load=-1):
    res = {}
    for name in os.listdir(path2photos):
        img = cv2.imread(os.path.join(path2photos, name))
        res[os.path.splitext(name)[0]] = img
        if max_photos2load > -1 and len(res) >= max_photos2load:
            break
    return res

def readAnnotation(fname):
    res = []
    with open(fname) as f:
        img_name = f.readline()
        for s in f:
            coords = map(float, s.split(','))
            res.append(coords)
    return img_name.rstrip(), np.array(res)

def readAnnotations(path2annotations):
    res = {}
    for name in os.listdir(path2annotations):
        img_name, lmarks = readAnnotation(os.path.join(path2annotations, name))
        res[img_name] = lmarks
    return res

def bboxFromLandmarks(landmarks, offset=0):
    x0 = 10000
    y0 = 10000
    x1 = 0
    y1 = 1
    for x, y in landmarks:
        if x < x0:
            x0 = x
        if x > x1:
            x1 = x
        if y < y0:
            y0 = y
        if y > y1:
            y1 = y
    return (x0 - offset/2.0, y0 - offset/2.0, x1 - x0 + offset, y1 - y0 + offset)

def drawLandmarks(img, landmarks):
    img = np.copy(img)
    for p in landmarks:
        cv2.circle(img, (int(p[0]), int(p[1])), 3, (0, 255, 0))
    return img

if __name__ == '__main__':
    path2photos      = '/home/daiver/coding/data/helen/all_photos/'
    path2annotations = '/home/daiver/coding/data/helen/annotation/'
    # path2cascade     = '/home/daiver/coding/jff/cv_data/cascades/frontalface10/haarcascade_frontalface_alt.xml'
    path2cascade     = '/home/daiver/coding/jff/cv_data/cascades/frontalface10/haarcascade_frontalface_default.xml'
    face_cascade = cv2.CascadeClassifier(path2cascade)

    images = readPhotos(path2photos, 50)
    annotations = readAnnotations(path2annotations)
    #tmp = raw_input()    
    # for name in annotations.keys():
    for name in images.keys():
        img = images[name]
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)

        landmarks = annotations[name]
        img = drawLandmarks(img, landmarks)
        faces = [bboxFromLandmarks(landmarks, 30)]
        for (x,y,w,h) in faces:
            center_x = x + w/2.0
            center_y = y + h/2.0
            w = int(w*1.0)
            h = int(h*1.0)
            x = int(center_x - w / 2)
            y = int(center_y - h / 2)
            cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
            # roi_gray = gray[y:y+h, x:x+w]
            # roi_color = img[y:y+h, x:x+w]
        cv2.imshow('', img)
        cv2.waitKey()
        #cv2.waitKey(1)
