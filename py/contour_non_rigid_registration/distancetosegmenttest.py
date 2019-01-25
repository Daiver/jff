import numpy as np
import loss

if __name__ == '__main__':
    import cv2
    p1 = np.array([0, 0])
    p2 = np.array([1, 0])
    p  = np.array([0, 1])

    print loss.projectPoint2Segment(p, p1, p2)

