import numpy as np
from scipy.interpolate import interp1d
from scipy import interpolate

#x = np.linspace(0, 10, 10)
#y = np.cos(-x**2/8.0)
x = np.linspace(0, 10, 5)
y = np.array([-1, 0.5, -1, -1, -1])
#x = np.linspace(0, 2, 2)
#y = np.array([-1, 2])

xnew = np.linspace(0, 10, 80)

f = interp1d(x, y)
#f2 = interp1d(x, y, kind='cubic')
f2 = interp1d(x, y, kind='cubic')(xnew)
#tck = interpolate.splrep(x, y, s=0)
s = interpolate.InterpolatedUnivariateSpline(x, y)
ynew = s(xnew)#interpolate.splev(xnew, tck, der=0)


import matplotlib.pyplot as plt
plt.plot(x,y,'o',xnew,f(xnew),'-', xnew, ynew,'--', xnew, f2, '+r')
plt.legend(['data', 'linear', 'spline', 'cubic'], loc='best')
plt.show()
