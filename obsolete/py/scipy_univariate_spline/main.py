import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import UnivariateSpline
nItems = 500
x = np.linspace(-3, 3, nItems)
y = np.exp(-x**2) + 0.1 * np.random.randn(nItems)
plt.plot(x, y, 'ro', ms=5)

smoothFactor = float(nItems) / 50.0

spl = UnivariateSpline(x, y, k = 2, s = smoothFactor)
xs = np.linspace(-3, 3, 1000)
plt.plot(xs, spl(xs), 'g', lw=3)

coeffs = spl.get_coeffs()
knots  = spl.get_knots()

print 'coeffs', coeffs
print 'knots', knots

knotsY = spl(knots)
plt.plot(knots, knotsY, 'bo')

#spl.set_smoothing_factor(0.5)
#print 'coeffs', spl.get_coeffs()
#print 'knots', spl.get_knots()
#plt.plot(xs, spl(xs), 'b', lw=3)
plt.show()

