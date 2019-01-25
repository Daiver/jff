import os
import cv2

import config
import common

def main():
    cropSomeFaces()

def cropSomeFaces():
    dirs = map(lambda x: os.path.join(config.original_dataset_path, x), 
            os.listdir(config.original_dataset_path))
    dirs_with_big_number_of_samples = filter(lambda x: len(os.listdir(x)) > 7, dirs)
    print len(dirs_with_big_number_of_samples)
    face_detector = cv2.CascadeClassifier(config.face_cascade_path)

    for d in dirs_with_big_number_of_samples:
        files = os.listdir(d)
        name = d.split(os.path.sep)[-1]
        out_dir_name = os.path.join(config.cropped_dataset_path, name)
        os.popen('mkdir -p %s' % out_dir_name).read()
        #print name
        for f in files:
            #print d, f
            img = cv2.imread(os.path.join(d, f))
            faces = face_detector.detectMultiScale(img)
            #print faces
            if len(faces) < 1:
                print f, 'have no faces'
            for i, fc in enumerate(faces):
                out_file_name = os.path.join(out_dir_name, '%s_%d.jpg' % (f, i))
                #cv2.imwrite(out_file_name, common.cutImage(img, fc))
                #x, y, w, h = fc
                #cv2.rectangle(img, (x, y), (x + w, h +y), (255, 0, 0))
            #cv2.imshow('', img)
            #cv2.waitKey()


if __name__ == '__main__':
    main()
