import cv2
import os

face_cascade = cv2.CascadeClassifier('/home/daiver/coding/libs/opencv/data/haarcascades/haarcascade_frontalface_default.xml')

dirName = 'data'
listOfImages = [os.path.join(dirName, x) for x in os.listdir(dirName)]

for name in listOfImages:
    img = cv2.imread(name)
    faces = face_cascade.detectMultiScale(img)
    for (x,y,w,h) in faces:
        cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)

    cv2.imshow('', img)
    cv2.waitKey()
