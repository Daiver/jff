import numpy as np
import matplotlib.pyplot as plt

#ys = np.array([
    #0.10000000000000,
    #0.20000000000000,
    #0.30000000000000,
    #0.22194627898983,
    #0.19114163802605,
    #0.17194568042104,
    #0.15936590180145,
    #0.15083965115733,
    #0.14493499235987,
    #0.14080014633822,
    #0.13790362510105,
    #0.13590093975779,
    #0.13456165968154,
    #0.13372740957785,
    #0.13328667625422,
    #0.13315918836120,
    #0.13328596087266,
    #0.13362279935504,
    #0.13413597409479,
    #0.13479928621596,
#])
#xs = np.arange(1, 21) / 10.0

#ys = np.array([

    #0.13315918836120 ,
    #0.13328596087266 ,
    #0.13362279935504 ,
    #0.13413597409478 ,
    #0.13479928621597 ,
    #0.13559204452373 ,
    #0.13649764869504 ,
    #0.13750258258625 ,
    #0.13859568898484 ,
    #0.13976764013233 
#])
#xs = np.arange(6, 26) / 10.0

ys = np.array([
    0.17194568042104 ,
    0.15936590180145 ,
    0.15083965115733 ,
    0.14493499235987 ,
    0.14080014633822 ,
    0.13790362510105 ,
    0.13590093975779 ,
    0.13456165968154 ,
    0.13372740957785 ,
    0.13328667625422 ,
    0.13315918836120 ,
    0.13328596087266 ,
    0.13362279935504 ,
    0.13413597409478 ,
    0.13479928621596 ,
    0.13559204452374 ,
    0.13649764869504 ,
    0.13750258258625 ,
    0.13859568898484 ,
    0.13976764013233 ,
    0.14101054617463 ,
    0.14231766197713 ,
    0.14368316491533 ,
    0.14510198446125 ,
    0.14656966998619 ,
    0.14808228706456 ,
    0.14963633525608 ,
    0.15122868224045 ,
    0.15285651052642 ,
    0.15451727392442 ,

    0.15620866167243 ,
    0.15792856861622 ,
    0.15967507022188 ,
    0.16144640147867 ,
    0.16324093895982 ,
    0.16505718546718 ,
    0.16689375680628 ,
    0.16874937033054 ,
    0.17062283496493 ,
    0.17251304247486 ,
    0.17441895978969 ,
    0.17633962222462 ,
    0.17827412747222 ,
    0.18022163025650 ,
    0.18218133756022 ,
    0.18415250435060 ,
    0.18613442973984 ,
    0.18812645352690 ,
    0.19012795307448 ,
    0.19213834048208 
])

xs = np.arange(6, 56) / 10.0

#xs = xs[27:]
#ys = ys[27:]

A = np.concatenate(( 
	    np.ones((xs.shape[0], 1)),
	    xs.reshape((-1, 1)),
	    (xs**2).reshape((-1, 1)),
	    (xs**3).reshape(-1, 1),
	    (xs**4).reshape(-1, 1)
            #np.exp(xs).reshape((-1, 1))
	    #(xs**5).reshape(-1, 1),
	    #(xs**6).reshape(-1, 1),
	    #(xs**7).reshape(-1, 1)
	), axis=1)
b = ys
print A.shape
coeffs = np.linalg.lstsq(A, b)[0]
print coeffs

ys2 = coeffs[0] + coeffs[1] * xs + coeffs[2]*(xs**2) + coeffs[3] * (xs**3) + coeffs[4] * (xs**4) #+ coeffs[5] * (xs**5) + coeffs[6] * (xs **6) + coeffs[7] * xs ** 7

plt.plot(xs, ys, 'ro')
#plt.plot(xs, ys, 'ro', xs, ys2, 'g--')
plt.show()