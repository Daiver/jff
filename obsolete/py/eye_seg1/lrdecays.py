import math

def expDecay(
        initial_lrate = 0.001, 
        drop = 0.5,
        epochs_drop = 20.0):
    def f(epoch):
        lrate = initial_lrate * math.pow(drop, math.floor((1+epoch)/epochs_drop))
        return lrate
    return f

def invSqrtWithLastStepsDecay(
        initial_lrate = 0.001, 
        start2DecayEpoch = 170,
        lastEpoch = 200):
    def f(epoch):
        if epoch < start2DecayEpoch:
            return initial_lrate * 1.0/np.sqrt(epoch + 1)
        valAt170 = initial_lrate * 1.0/np.sqrt(epoch + 1)
        return (lastEpoch - epoch) / (lastEpoch - start2DecayEpoch) * valAt170
    return f

def invSqrtDecay(
        initial_lrate = 0.001):
    def f(epoch):
        return initial_lrate * 1.0/np.sqrt(epoch + 1)
    return f

def linearDecay(
        initial_lrate = 0.001,
        nIters = 200):
    nIters = float(nIters)
    def f(epoch):
        return initial_lrate * (1.0 - epoch / nIters)
    return f

def linearWithRampUp(
        lrate_before_rampup = 0.001,
        initial_lrate = 0.001,
        itersForRampUp = 5,
        nIters = 200):
    nIters = float(nIters - itersForRampUp)
    def f(epoch):
        if epoch < itersForRampUp:
            deltaLr = initial_lrate - lrate_before_rampup
            return epoch / float(itersForRampUp) * deltaLr + lrate_before_rampup
        return initial_lrate * (1.0 - (epoch - itersForRampUp) / nIters)
    return f

def sqrtDecay(
        initial_lrate = 0.001,
        nIters = 200):
    nIters = float(nIters)
    def f(epoch):
        return initial_lrate * math.sqrt(1.0 - epoch / nIters)
    return f

