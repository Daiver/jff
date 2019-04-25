import numpy as np

class PolynomApproximator:
    def __init__(self, points, weights = None, power = 1, regCoeff = 0.0):
        self.power = power
        self.coeffs = PolynomApproximator.fitCurve(points, weights, power, regCoeff)

    @staticmethod
    def xFeatsToPolyFeats(X, power=1):
        assert len(X.shape) == 1
        nSamples = X.shape[0]
        res = np.zeros((nSamples, power + 1), dtype=np.float32)
        for i, x in enumerate(X):
            for j in xrange(power):
                res[i, j] = x**(j + 1)

        res[:, -1] = 1.0
        return res

    @staticmethod
    def fitCurve(points, weights=None, power=1, regCoeff = 0.0):
        if weights is None:
            weights = np.ones(len(points))
        W = np.vstack([weights] * (power + 1)).T
        A = PolynomApproximator.xFeatsToPolyFeats(points[:, 0], power)
        A *= W
        b = points[:, 1]
        b = b * weights
        AtA = A.T.dot(A)
        Atb = A.T.dot(b)
        AtAReg = AtA + np.eye(AtA.shape[0]) * regCoeff
        return np.linalg.solve(AtAReg, Atb)

    def __call__(self, x):
        return PolynomApproximator.xFeatsToPolyFeats(x, self.power).dot(self.coeffs)



