import numpy as np
import h5py
import cv2

def dumpData(img, i, phisT, bboxesT):
    cv2.imwrite('dump/' + str(i) + '.png', img)
    with open('dump/' + str(i) + '.txt', 'w') as f:
        f.write('%s %s %s %s\n' % (
            str(bboxesT[0][i]),
            str(bboxesT[1][i]),
            str(bboxesT[2][i]),
            str(bboxesT[3][i])))
        for j in xrange(29):
            f.write('%s %s\n' % (str(phisT[j][i]), str(phisT[j + 29][i])))

if __name__ == '__main__':
    fl = h5py.File('./COFW_test.mat')
    print fl.keys()
    imgRef = fl['IsT']
    bboxesT = fl['bboxesT']
    #phisT
    phisT = fl['phisT']
    print np.array(phisT).shape
    i = 0
    #for i, x in enumerate(imgRef[0]):
    featInd = 16
    while True:
        x = imgRef[0][i]
        img = np.array(fl[x]).T
        cv2.imwrite('tmp.png', img)  
        img = cv2.imread('tmp.png', 0)
        cv2.rectangle(img, 
                (int(bboxesT[0][i]), int(bboxesT[1][i])), 
                (int(bboxesT[0][i] + bboxesT[2][i]), 
                                     int(bboxesT[1][i] + bboxesT[3][i])), 255)
        print 'Point', int(phisT[featInd][i]), int(phisT[29 + featInd][i])
        cv2.circle(img, (int(phisT[featInd][i]), int(phisT[29 + featInd][i])), 5, 255)
        cv2.imshow('', img)
        print 'ImgId', i, 'featInd', featInd
        #print phisT[i][0], phisT[i][0 + 29]
        key = cv2.waitKey() % 0x100
        print key
        if key == 27:
            exit()
        elif key == 81:
            i -= 1
        elif key == 83:
            i += 1
        elif key == 82:
            featInd -= 1
        elif key == 84:
            featInd += 1
        elif key == 32:
            dumpData(np.array(fl[x]).T, i, phisT, bboxesT)
        
        cv2.destroyAllWindows()
        #cv2.imwrite('tmp.png', img)  
        #print img
        #cv2.imwrite('./testImages/%s.png' % str(i), img)
    #img = np.array(fl[imgRef[0][6]]).T
    #print img.shape
    #cv2.imwrite('tmp.png', img)
    #img = cv2.imread('tmp.png', 0)

