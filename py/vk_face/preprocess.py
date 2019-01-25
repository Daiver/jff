import os
import numpy as np
import cv2

face_cascade_path = os.path.join(os.path.dirname(__file__), "frontalface10/haarcascade_frontalface_alt.xml")
eye_cascade_path = os.path.join(os.path.dirname(__file__), "frontalface10/haarcascade_eye_tree_eyeglasses.xml")

DOWNSCALE = 1
face_detector = cv2.CascadeClassifier(face_cascade_path)
eye_detector = cv2.CascadeClassifier(eye_cascade_path)

def main():
    dumps_dir = '/home/daiver/dumps/vk_face/'
    total = 0
    for dir_name in os.listdir(dumps_dir):
        os.system('mkdir %s' % os.path.join(dumps_dir, dir_name, 'faces'))
        for name in filter(lambda x: x[-4:] == '.jpg', os.listdir(os.path.join(dumps_dir, dir_name, 'raw'))):
            fname = os.path.join(dumps_dir, dir_name, 'raw', name)
            print fname
            img = cv2.imread(fname)
            if img == None:continue
            faces = detect_faces(img)
            img2 = img.copy()
            total += 1

            for i, f in enumerate(faces):
                x, y, w, h = [ v*DOWNSCALE for v in f ]
                path_to_write = os.path.join(dumps_dir, dir_name, 'faces', '%s_%d.jpg' % (name, i))
                face = cutRect(img2, (x, y, w, h))
                eyes = eye_detector.detectMultiScale(face)
                print eyes
                #for e in eyes:
                #    x1, y1, w1, h1 = e
                #    cv2.rectangle(face, (x1,y1), (x1+w1,y1+h1), (0,255,0), 3)
                cv2.imwrite(path_to_write, face)

                #cv2.imshow(str(i), face)
                #cv2.rectangle(img, (x,y), (x+w,y+h), (0,255,0), 3)
            #cv2.imshow('', img)
            #cv2.waitKey()
    print total

def cutRect(img, rect):
    x, y, w, h = rect
    return img[y:y+h, x:x+w]

def detect_faces(img):
    minisize = (img.shape[1]/DOWNSCALE,img.shape[0]/DOWNSCALE)
    miniframe = cv2.resize(img, minisize)
    faces = face_detector.detectMultiScale(miniframe)
    return faces
    
        
    '''file_name = fname.split(os.path.sep)[-1]
    img2 = img.copy()
    res = []
    for i, f in enumerate(faces):
        x, y, w, h = [ v*DOWNSCALE for v in f ]
        small = img[y:y + h,x:x + w]
        face2 = face_detector.detectMultiScale(small)
        name = os.path.join(dir_name, '_'.join([file_name, str(i), '.png']))
        if len(face2) > 0: 
            face2 = face2[0]
            res.append((name, ((x + face2[0],y + face2[1]), (x + face2[0] + face2[2],y + face2[1] + face2[3]))))
            small = small[face2[1]:face2[3] + face2[1], face2[0]: face2[2] + face2[0]]
            cv2.rectangle(img2, (x + face2[0],y + face2[1]), (x + face2[0] + face2[2],y + face2[1] + face2[3]), (0,255,0), 3)
        else:
            res.append((name, ((x,y), (x+w,y+h))))
            cv2.rectangle(img2, (x,y), (x+w,y+h), (0,255,0), 3)
        cv2.imwrite(name, small)

    cv2.imwrite(fname + '.face_marked.jpg', img2)
    return res'''
 


if __name__ == '__main__':
    main()
