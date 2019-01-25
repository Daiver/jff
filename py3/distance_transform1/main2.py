import numpy as np
np.set_printoptions(edgeitems=50, linewidth=175)
import matplotlib.pyplot as plt
import cv2
import scipy
from scipy.optimize import minimize

'''
cv::Vec3b getColorSubpix(const cv::Mat& img, cv::Point2f pt)
{
    assert(!img.empty());
    assert(img.channels() == 3);

    int x = (int)pt.x;
    int y = (int)pt.y;

    int x0 = cv::borderInterpolate(x,   img.cols, cv::BORDER_REFLECT_101);
    int x1 = cv::borderInterpolate(x+1, img.cols, cv::BORDER_REFLECT_101);
    int y0 = cv::borderInterpolate(y,   img.rows, cv::BORDER_REFLECT_101);
    int y1 = cv::borderInterpolate(y+1, img.rows, cv::BORDER_REFLECT_101);

    float a = pt.x - (float)x;
    float c = pt.y - (float)y;

    uchar b = (uchar)cvRound((img.at<cv::Vec3b>(y0, x0)[0] * (1.f - a) + img.at<cv::Vec3b>(y0, x1)[0] * a) * (1.f - c)
                           + (img.at<cv::Vec3b>(y1, x0)[0] * (1.f - a) + img.at<cv::Vec3b>(y1, x1)[0] * a) * c);
    uchar g = (uchar)cvRound((img.at<cv::Vec3b>(y0, x0)[1] * (1.f - a) + img.at<cv::Vec3b>(y0, x1)[1] * a) * (1.f - c)
                           + (img.at<cv::Vec3b>(y1, x0)[1] * (1.f - a) + img.at<cv::Vec3b>(y1, x1)[1] * a) * c);
    uchar r = (uchar)cvRound((img.at<cv::Vec3b>(y0, x0)[2] * (1.f - a) + img.at<cv::Vec3b>(y0, x1)[2] * a) * (1.f - c)
                           + (img.at<cv::Vec3b>(y1, x0)[2] * (1.f - a) + img.at<cv::Vec3b>(y1, x1)[2] * a) * c);

    return cv::Vec3b(b, g, r);
}
'''

#SUPER SLOW
def takePixelInterpolated(img, pt):
    rows, cols = img.shape
    y, x = pt

    x0 = cv2.borderInterpolate(int((x)),   cols, cv2.BORDER_REFLECT_101);
    x1 = cv2.borderInterpolate(int((x+1)), cols, cv2.BORDER_REFLECT_101);
    y0 = cv2.borderInterpolate(int((y)),   rows, cv2.BORDER_REFLECT_101);
    y1 = cv2.borderInterpolate(int((y+1)), rows, cv2.BORDER_REFLECT_101);

    a = x - int(x)
    c = y - int(y)
    b = ((img[y0, x0] * (1.0 - a) + img[y0, x1] * a) * (1.0 - c) + (img[y1, x0] * (1.0 - a) + img[y1, x1] * a) * c)
    return b

#TODO: ADD CONSTANT FOLDING
def discretPoints2ContourLoss(points, contourImg):
    #distField = cv2.distanceTransform(img, cv2.DIST_L1, 0)
    distField = cv2.distanceTransform(contourImg, cv2.DIST_L2, 0)
    return sum(takePixelInterpolated(distField, p) for p in points)
    #return sum(distField[p[0], p[1]] for p in points.round().astype(np.uint32))

def discretPoints2ContourGrad(points, contourImg):
    distField = cv2.distanceTransform(contourImg, cv2.DIST_L2, 0)
    dxKernel = np.array([
        [-1, 0, 1],
        [-1, 0, 1],
        [-1, 0, 1]
    ], dtype=np.float32) / 9.0
    dx = cv2.filter2D(distField, -1, dxKernel)

    dyKernel = np.array([
        [-1, -1, -1],
        [0, 0, 0],
        [1, 1, 1]
    ], dtype=np.float32) / 9.0
    dy = cv2.filter2D(distField, -1, dyKernel)
    
    res = np.zeros_like(points)
    for i, p in enumerate(points):
        res[i, 0] = takePixelInterpolated(dx, p)
        res[i, 1] = takePixelInterpolated(dy, p)
    print('p>', points)
    print('g>', res)
    return res

def main():
    img = np.zeros((7, 7), dtype=np.uint8)
    img[:] = 1
    img[2, 2] = 0
    img[3, 3] = 0
    #img[4, 3] = 0
    print(img)

    points = np.array([
        [5, 5], 
    ])

    print('dist', discretPoints2ContourLoss(points, img)) 

    loss = lambda x: discretPoints2ContourLoss(x.reshape((-1, 2)), img)
    def numJac(x):
        print('x>', x)
        #AGAIN, SUPER SLOW
        dx = 0.3
        #dx = 0.51
        #dx = 1.0
        res = np.zeros(x.shape)
        fx = loss(x)
        for i in range(x.shape[0]):
            x1 = np.copy(x)
            x1[i] -= dx
            fx1 = loss(x1)
            x1[i] += 2*dx
            fx2 = loss(x1)
            res[i] = (fx2 - fx1) / dx / 2.0
            #print(fx1, fx2, x1)
        print('g>', res)
        return res

    def discretJac(x):
        return discretPoints2ContourGrad(x.reshape((-1, 2)), img).reshape(-1)

    #jac = numJac
    #jac = None
    jac = discretJac

    optRes = minimize(loss, points.reshape(-1), jac=jac)
    print(optRes)
    print('x>', optRes['x'].round())
    


if __name__ == '__main__':
    main() 
   
