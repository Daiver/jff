import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
import sys
import reader
import horn

if __name__ == '__main__':
    execFileName = os.path.realpath(__file__)
    pathToBinary = os.path.join(os.path.dirname(execFileName), 'tvl1flow_3', 'tvl1flow')

    onlyRead = False

    i0Name = sys.argv[1]
    if not onlyRead:
        i1Name = sys.argv[2]
        command = '%s %s %s %s' % (pathToBinary, i0Name, i1Name, 'flow.flo')
        #command = '%s %s %s %s 0 0.25 0.15 0.3 100 0.5 5 0.001 1' % (pathToBinary, i0Name, i1Name, 'flow.flo')
        print command
        os.system(command)

    u, v = reader.readFlow('flow.flo')
    img0 = cv2.imread(i0Name)

    img1 = cv2.imread(i1Name)
    res = horn.translate2(img0, u, v)
    cv2.imwrite("res.png", res)
    cv2.imshow('', res)
    print sum(sum(abs(img0 - res))) / float(img1.size)
    cv2.waitKey()
    
    u, v = cv2.pyrDown(u), cv2.pyrDown(v)
    u, v = cv2.pyrDown(u), cv2.pyrDown(v)
    #v *= -1

    plt.gca().invert_yaxis()

    plt.quiver(u, v, u**2 + v**2)
    plt.show()
    
