import cv2
import numpy as np
import sys

if __name__ == '__main__':
    name = sys.argv[1]
    img = cv2.imread(name)
    
    res = np.zeros_like(img)
    res[273:405, 45:175] = img[273:405, 45:175]
    #res = cv2.Laplacian(res, 5)
    cv2.imwrite(sys.argv[2], res)
