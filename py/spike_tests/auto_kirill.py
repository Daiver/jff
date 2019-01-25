import sys
import numpy as np
import matplotlib.pyplot as plt
import deficiency_angle

def defAngles(xs):
    v1 = [0.0, 0.0, 0.0]
    v2 = [2.0, 0.0, 0.0]
    v3 = [2.0, 0.0, 2.0]
    v4 = [0.0, 0.0, 2.0]
    v5 = [1.0, 0.0, 1.0]

    res = []
    for x in xs:
        v = v5
        v[1] = x
        res.append(deficiency_angle.defficiencyAngle(v, [v1, v2, v3, v4]))
    return np.array(res)

def extractSignificantVerticesFromObj(fname):
    res = []
    verticesCounter = 0
    with open(fname) as f:
        for s in f:
            if s[0] != 'v':
                continue
            if (verticesCounter - 1) % 3 == 0 and verticesCounter > 1:
                res.append(float(s.split(' ')[2]))
            verticesCounter += 1
    return np.array(res)

if __name__ == '__main__':
    #ys = extractSignificantVerticesFromObj(sys.argv[1])
    ys = extractSignificantVerticesFromObj('./auto_slavik3_s0.obj')
    #ys2 = extractSignificantVerticesFromObj('./auto_slavik4_s0.obj')
    ys2 = extractSignificantVerticesFromObj('./auto_slavik3_s10.obj')
    ys3 = extractSignificantVerticesFromObj('./auto_slavik3_s20.obj')
    ys4 = extractSignificantVerticesFromObj('./auto_slavik3_s30.obj')
    print ys.shape
    print np.min(ys), np.max(ys)
    #print ys
    nSamples = ys.shape[0]
    xs = np.linspace(0.3, 0.3 + 0.01 * (nSamples - 1), nSamples)
    #xs = xs[:900]
    #ys = ys[:900]
    xs2 = defAngles(xs)
    #A = np.concatenate((
        #np.ones((xs2.shape[0], 1)),
        #xs2.reshape((-1, 1)),
        #(xs2**2).reshape((-1, 1)),
        #(xs2**3).reshape((-1, 1)),
        #(xs2**4).reshape((-1, 1)),
        #), axis=1)
    #b = ys
    #coeffs = np.linalg.lstsq(A, b)[0]
    #print coeffs
    #ys2 = (coeffs[0] 
         #+ xs2 * coeffs[1] 
         #+ (xs2 ** 2) * coeffs[2]
         #+ (xs2 ** 3) * coeffs[3]
         #+ (xs2 ** 4) * coeffs[4]
           #)

    #ys /= xs2
    #plt.plot(xs, ys, 'go')#, xs, ys3, 'bo', xs, ys4, 'yo')
    plt.plot(xs2, ys, 'go')#, xs, ys3, 'bo', xs, ys4, 'yo')
    #plt.plot(xs2, ys, 'ro', xs2, ys2, 'g--')#, xs, ys3, 'bo', xs, ys4, 'yo')
    #plt.plot(xs, ys, 'ro', xs, ys2, 'go', xs, ys3, 'bo', xs, ys4, 'yo')
    #plt.plot(xs2, ys, 'ro', xs2, ys2, 'go', xs2, ys3, 'bo', xs2, ys4, 'yo')
    plt.show()

