import math

def dotproduct(v1, v2): return sum((a*b) for a, b in zip(v1, v2))

def length(v): return math.sqrt(dotproduct(v, v))

def angleT(v1, v2): 
    return math.acos(dotproduct(v1, v2) / (length(v1) * length(v2)))

def sub(v1, v2):
    return [x - y for x, y in zip(v1, v2)]

def defficiencyAngle(center, verticesByOrder):
    res = 0.0
    for i in xrange(len(verticesByOrder)):
        v1 = verticesByOrder[i]
        v2 = verticesByOrder[(i + 1) % len(verticesByOrder)]
        res += angleT(sub(v1,center), sub(v2, center))
    return math.pi * 2 - res

if __name__ == '__main__':

    #v1 = [0.0, 0.0, 0.0]
    #v2 = [2.0, 0.0, 0.0]
    #v3 = [2.0, 0.5, 0.0]
    #v4 = [0.0, 0.5, 0.0]
    #v5 = [1.0, 0.25, 0.155]

    v1 = [0.0, 0.0, 0.0]
    v2 = [2.0, 0.0, 0.0]
    v3 = [2.0, 2.0, 0.0]
    v4 = [0.0, 2.0, 0.0]
    v5 = [1.0, 1.0, 0.3]

#    import matplotlib.pyplot as plt
    #import numpy as np
    #offset = 0.3
    #step = 0.01
    #ys = []
    #for i in xrange(150):
        #v = v5
        #v[2] = offset + i * step
        #ys.append(defficiencyAngle(v, [v1, v2, v3, v4]))

    #plt.plot(np.linspace(0.3, 0.03 + 0.01 * 149, 150), ys, 'ro')
    #plt.show()

    #v1 = [0.0, 0.0, 0.0]
    #v2 = [2.0, 0.0, 0.0]
    #v3 = [2.0, 1.0, 0.0]
    #v4 = [0.0, 1.0, 0.0]
    #v5 = [1.0, 0.5, 0.02]

    #borders for 0 smooth
    #allowed     0.174231563703 
    #looks like 10' is border
    #not allowed 0.174618577226 

    #borders for 100 smooth
    #allowed     0.00179828173151 
    #not allowed 0.00192004114709
    print defficiencyAngle(v5, [v1, v2, v3, v4]) #/ math.pi*180.0
    print defficiencyAngle(v5, [v1, v2, v3, v4]) / math.pi*180.0

    #print defficiencyAngle(v1, [v2, v5, v4]) #/ math.pi*180.0
    # 3.12198355381 
