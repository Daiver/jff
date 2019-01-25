import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

def dot(v1, v2):
    return tf.reduce_sum(v1 * v2)

def predict(w, data):
    return [1 if w.dot(x) > 0 else -1 for x in data]

def predictErr(w, data, labels):
    return len(labels) - sum(np.array(labels, dtype=np.int) == np.array(predict(w, data), dtype=np.int))

def trainSVM_TF(wInit, data, labels, clambda, nIters):
    nSamples, nFeatures = data.shape
    w = tf.Variable(np.array(wInit, np.float32))
    y = tf.placeholder("float")
    X = tf.placeholder("float", nFeatures)

    errExp = clambda * dot(w[:nFeatures - 1], w[:nFeatures - 1]) + tf.maximum(0.0, 1.0 - y * dot(w, X))
    trainExp = tf.train.GradientDescentOptimizer(0.01).minimize(errExp)

    sess = tf.Session()
    init = tf.initialize_all_variables()
    sess.run(init)

    for iterInd in xrange(nIters):
        sampleInd = np.random.randint(0, nSamples)
        sess.run(trainExp, feed_dict = {X: data[sampleInd], y: float(labels[sampleInd])})
        err = predictErr(sess.run(w), data, labels)

        if err == 0:
            break

        if iterInd % 100 == 0:
            print iterInd, err 
    wVal = sess.run(w)
    return wVal


def trainSVM_Manual(wInit, data, labels, clambda, nIters):
    nSamples, nFeatures = data.shape
    w = np.array(wInit, dtype=np.float32)
    grad = np.zeros(nFeatures)
    step = 0.01
    for iterInd in xrange(nIters):
        sampleInd = np.random.randint(0, nSamples)
        x = data[sampleInd, :] 
        y = float(labels[sampleInd])
        #errExp = clambda * dot(w[:nFeatures - 1], w[:nFeatures - 1]) + tf.maximum(0.0, 1.0 - y * dot(w, X))
        grad[:-1] = clambda * 2.0 * w[:-1]
        grad[-1]  = 0.0
        if 1.0 - y * w.dot(x) >= 0.0:
            grad += - y * x
        w -= grad * step
        err = predictErr(w, data, labels)

        if err == 0:
            break

        if iterInd % 100 == 0:
            print iterInd, err 
    return w

if __name__ == '__main__':

    nSamples = 1000
    data = np.array([[np.random.uniform(-3, 3), np.random.uniform(-3, 3), 1] for i in xrange(nSamples)])
    labels = np.array([1 if x[1] >= 0 else -1 for x in data], dtype=np.int)
    # labels = np.array([1 if x[0] * -0.3 + x[1] * -1 - 0.3 >= 0 else -1 for x in data], dtype=np.int)

    nSamples, nFeatures = data.shape
    nIters = nSamples * 2

    data += np.random.normal(0.3, 1.0, [nSamples, nFeatures])
    
    clambda = 0.1
    wInit = np.array([1, 0.0, 0.0], dtype=np.float32)
    #wInit = np.ones(nFeatures)
    
    wVal = trainSVM_TF(wInit, data, labels, clambda, nIters)
    #wVal = trainSVM_Manual(wInit, data, labels, clambda, nIters)

    svmDirection, svmIntercept = wVal[0:2], wVal[2]
    svmDirection /= np.linalg.norm(svmDirection)
    print 'direction', svmDirection, 'intercept', svmIntercept

    pos = labels ==  1
    neg = labels == -1
    xsPos, ysPos = data[pos, 0], data[pos, 1]
    xsNeg, ysNeg = data[neg, 0], data[neg, 1]

    xsSvm = np.linspace(np.min(data[:, 0]), np.max(data[:, 1]))
    ysSvm = np.array([-(x * svmDirection[0] + svmIntercept) / svmDirection[1] for x in xsSvm])

    plt.plot(xsPos, ysPos, 'xg', xsNeg, ysNeg, 'xr', xsSvm, ysSvm, 'b--')
    plt.show()
