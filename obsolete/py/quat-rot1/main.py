import numpy as np

#quat : [u, v, w, s]
def quatRotMatrix(quat):
    u, v, w, s = quat
    return np.array([
        [s**2 + u**2 - v**2 - w**2, 2 * (u*v - s*w),           2 * (u*w + s*v)          ],
        [2 * (u*v + s*w),           s**2 - u**2 + v**2 - w**2, 2 * (v*w - s*u)          ],
        [2 * (u*w - s*v),           2 * (v*w + s*u),           s**2 - u**2 - v**2 + w**2]
        ], dtype=np.float32)

def rotateByQuat(quat, point):
    return np.dot(((1.0/(np.dot(quat, quat))) * quatRotMatrix(quat)), point)

def jacobiFromRotationWithQi(point):
    x, y, z = point
    return np.array([
        [ 0.0,  2*z, -2*y],
        [-2*z,  0.0,  2*x],
        [ 2*y, -2*x,  0.0]
        ], dtype=np.float32)


