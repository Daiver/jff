import numpy as np
from scipy.optimize import minimize
import input_data
from sklearn.preprocessing import normalize
#from linesearch import gradDescent
from  nonlinear_cg import nonLinCG

def sigmoid(z):
    #print z
    return 1.0/(1.0 + np.exp(-z))


def logFN(theta, x):
    return sigmoid(theta[:-1].dot(x) + theta[-1])

def massLogFN(theta, xs):
    return np.array([logFN(theta, x) for x in xs])

def getJ(X, Y):
    def res(theta):
        sm = 0.0
        act = (X.dot(theta[:-1])) + theta[-1]
        f   = 1.0/(1.0 + np.exp(-act))
        return -np.sum(Y * np.log(f) + (1.0 - Y) * np.log(1.0 - f))
#        for i in xrange(X.shape[0]):
            #x = X[i]
            #y = Y[i]
            #fnVal = logFN(theta, x)
            #sm += y * np.log(fnVal) + (1.0 - y) * np.log(1.0 - fnVal)
        #return -sm
    return res
#    return lambda theta: -sum([
        #y*np.log(logFN(theta, x)) + (1.0 - y)*np.log(1.0 - logFN(theta, x))
        #for x, y in zip(X, Y)])

def getGrad(X, Y):
    def res(theta):
        tmp = np.zeros(X.shape[1] + 1)
#        act = (X.dot(theta[:-1])) + theta[-1]
        #f   = 1.0/(1.0 + np.exp(-act))
        #diff = Y - f
        #tmp[:X.shape[1]] = X.T.dot(diff)
        #tmp[-1] = sum(diff)
        #return -tmp
        for i in xrange(X.shape[0]):
            y = Y[i]
            x = X[i]
            diff = (y - logFN(theta, x))
            tmp[:X.shape[1]] += (diff) * x
            tmp[ X.shape[1]] += (diff)

        return -tmp    

    return res

#X = np.array([0.50, 0.75, 1.00, 1.25, 1.50, 1.75, 1.75, 2.00, 2.25, 2.50, 2.75, 3.00, 3.25, 3.50, 4.00, 4.25, 4.50, 4.75, 5.00, 5.50]).reshape((-1, 1))
#Y = np.array([0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 1, 1, 1, 1, 1])
#print X.shape, Y.shape

#J = getJ(X, Y)
#gradJ = getGrad(X, Y)

#res = minimize(J, np.array([0.0, 0.0]), 
       ###method='Newton-CG', 
       ##method='bfgs', 
       #method='nelder-mead', 
       #options={'disp': True})
#print 'theta', res

##print J(res['x'])

#theta = np.array([0.01, 0.02])
#dx = 0.000001
#gr = np.zeros(2)
#for j in xrange(theta.shape[0]):
    #gr[j] -= J(theta)
    #theta[j] += dx
    #gr[j] += J(theta)
    #theta[j] -= dx
    #gr[j] /= dx

#print gr
#print gradJ(theta)

#print gradDescent(J, gradJ, np.array([0.02, 0.01]), 1000)

def train():
    mnist = input_data.read_data_sets("/home/daiver/Downloads/MNIST_data/", one_hot=True)
    print (mnist.train.next_batch(100)[0].shape)
    print (mnist.train.next_batch(100)[1].shape)

    thetaShape = 784 + 1

    thetas = []

    for labelInd in xrange(10):
        theta = np.random.rand(thetaShape) * 0.01
        #print mnist.num_examples
        batch = mnist.train.next_batch(55000)
        X = batch[0]
        X = normalize(X)
        Y = batch[1][:, labelInd]
        J = getJ(X, Y)
        gradJ = getGrad(X, Y)
        res = nonLinCG(J, gradJ, theta, 200)
        #res = gradDescent(J, gradJ, theta, 2000)
        theta = res
        #res = minimize(J, np.random.rand(thetaShape), 
               ##method='Newton-CG', 
               ##method='bfgs', 
               #method='nelder-mead', 
               #options={'disp': True})
        #print 'theta', res
        print labelInd, J(res)
        thetas.append(theta)

    np.save("mnist_class4.dump", np.array(thetas))

def test():
    mnist = input_data.read_data_sets("/home/daiver/Downloads/MNIST_data/", one_hot=True)
    X = mnist.test.images
    X = normalize(X)
    Y = mnist.test.labels
    #thetas = np.load("./mnist_class.dump.npy")
    thetas = np.load("./mnist_class4.dump.npy")
    print thetas.shape
    nSamples = 10000
    err = 0
    for i in xrange(nSamples):
        ansInd = np.argmax(Y[i])
        x = X[i]
        values = np.array([logFN(t, x) for t in thetas])
        res = np.argmax(values)
        if res != ansInd:
            err += 1
    print 1.0 - float(err)/nSamples, err, nSamples

if __name__ == "__main__":
    #train()
    test()
