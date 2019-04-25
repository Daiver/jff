import cvxpy
import numpy as np
import matplotlib.pyplot as plt
 
def varsToLinear(a, b):
    assert(a.shape[0] == 2)
    an = -a[0]/a[1] 
    bn = -b/a[1]
    return an[0, 0], bn[0, 0]

def draw(a, b, data, labels):
    xs = data[:, 0]
    ys1 = data[:, 1]
    ys2 = [a*x + b for x in xs]
    #plt.plot(xs, ys1, 'ro')
    xsP = []
    xsN = []
    ysP = []
    ysN = []
    for i, l in enumerate(labels):
        if l == 1:
            xsP.append(xs[i])
            ysP.append(data[i, 1])
        else:
            xsN.append(xs[i])
            ysN.append(data[i, 1])

    plt.plot(xsP, ysP, 'ro', xsN, ysN, 'go', xs, ys2)
    #plt.plot(xs, ys1, 'ro', xs, ys2)
    plt.show()

def createVars(nDims):
    return [cvxpy.Variable(nDims), cvxpy.Variable()]

def createConstraints(a, b, data, labels):
    assert(data.shape[1] == a.size[0])
    constraints = []
    for x, l in zip(data, labels):
        if l == 1:
            constraints.append(a.T*x + b >= 1)
        else:
            constraints.append(a.T*x + b <= -1)
    return constraints

def test(a, b, data, labels):
    nErr = 0
    for x, l in zip(data, labels):
        res = a.T.dot(x) + b
        if res < 1-0.000001 and l == 1:
            nErr += 1
            print 'ERR::'
        if res > -1+0.00001 and l == -1:
            nErr += 1
            print 'ERR::'
        print l, res
    print nErr, len(data)

if __name__ == '__main__':
    data = np.array([
        [1, 3],
        [1.1, 5],
        [1.1, 1.5],
        [5, 7],
        [4, 5],
        [3, 3.2],
        [2, 2.0],
        [5, 3],
        [3, 1],
        [1.8, 1.5]
        ])
    labels = np.array([1, 1, 1, 1, 1, 1, -1, -1, -1, -1])
    a, b = createVars(2)
    constraints = createConstraints(a, b, data, labels)

    obj = cvxpy.Minimize(cvxpy.norm2(a))
    prob = cvxpy.Problem(obj, constraints)
    print prob.solve()
    print 'A'
    print a.value
    print 'B', b.value

    test(a.value, b.value, data, labels)
    an, bn = varsToLinear(a.value, b.value)
    draw(an, bn, data, labels)
    #draw(0.01699999,  0.00999999, data, labels)

#from cvxpy import *

## Create two scalar optimization variables.
#x = Variable()
#y = Variable()

## Create two constraints.
#constraints = [x + y == 1,
               #-x + y >= 1]

## Form objective.
##obj = Minimize(square(x - y))
#obj = Minimize(square(x))

## Form and solve problem.
#prob = Problem(obj, constraints)
#prob.solve()

## The optimal dual variable (Lagrange multiplier) for
## a constraint is stored in constraint.dual_value.
#print "optimal (x + y == 1) dual variable", constraints[0].dual_value
#print "optimal (x - y >= 1) dual variable", constraints[1].dual_value
#print "x - y value:", (x - y).value
#print x.value
#print y.value
