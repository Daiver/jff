from __future__ import division
import cv2
import matplotlib.pyplot as plt
import numpy as np
from scipy import interpolate


nodes = np.array( [ [1, 2], [6, 15], [10, 6], [10, 3], [3, 7] ] )

x = nodes[:,0]
y = nodes[:,1]

tck,u     = interpolate.splprep( [x,y] ,s = 0 )
xnew,ynew = interpolate.splev( np.linspace( 0, 1, 100 ), tck,der = 0)

print xnew.shape
print ynew.shape

canvas = np.zeros((1000, 1000, 3), np.uint8) + 255
pts = np.zeros((xnew.shape[0], 2))
pts[:, 0] = xnew * 50
pts[:, 1] = ynew * 50
pts = pts.reshape((1, -1, 2)).astype(np.int32)
print pts
cv2.polylines(canvas, pts, False, (0, 0, 0), 2)
cv2.imshow('', canvas)
cv2.waitKey()

plt.plot( x,y,'o' , xnew ,ynew )
plt.legend( [ 'data' , 'spline'] )
plt.axis( [ x.min() - 1 , x.max() + 1 , y.min() - 1 , y.max() + 2 ] )
plt.show()
