import cv2
import numpy as np
import sys
#import scipy.interpolate
#import reader


if __name__ == '__main__':
    #img1 = cv2.imread('/home/daiver/1.png', 0)
    #img2 = cv2.imread('/home/daiver/2.png', 0)
    #img1 = cv2.imread('/home/daiver/Downloads/scene1.row3.col3.ppm', 0)
    #img2 = cv2.imread('/home/daiver/Downloads/scene1.row3.col5.ppm', 0)
    '''np.set_printoptions(threshold='nan')

    img1 = np.array([
            #[0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0], 
            [0, 128, 255, 128, 0, 0],
            [0, 0, 0, 0, 0, 0] 
            #[0, 0, 0, 0]
        ])
    img2 = np.array([
            #[0, 0, 0, 0],
            [0, 128, 255, 128, 0, 0],
            [0, 0, 0, 0, 0, 0] ,
            [0, 0, 0, 0, 0, 0] 
            #[0, 0, 0, 0]
        ])'''
    #img1 = img1.reshape((4, 1))
    #img2 = img2.reshape((4, 1))
    img1 = cv2.imread(sys.argv[1],0)
    img2 = cv2.imread(sys.argv[2],0)
    print img1.shape, img2.shape

    flow = cv2.calcOpticalFlowFarneback(img1, img2, 
            0.5, 5, 2, 10, 3, 0.7, cv2.OPTFLOW_USE_INITIAL_FLOW)
    print flow.shape
    #print flow
    u, v = cv2.split(flow)
    '''print 'u'
    print u
    print 'v'
    print v'''
    '''f = scipy.interpolate.interp2d(
            np.arange(img1.shape[0]),
            np.arange(img1.shape[0]),
            img1)'''
    #plotArrows(u, v)
