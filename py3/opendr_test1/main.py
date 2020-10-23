import numpy as np
# import matplotlib.pyplot as plt
import cv2
import opendr
from opendr.simple import *


"""
demo('texture')
demo('moments')
demo('per_face_normals')
demo('silhouette')
demo('boundary')
demo('point_light')
demo('spherical_harmonics')
demo('optimization')
demo('optimization_cpl')
"""


def main():
    # opendr.demo("optimization")
    w, h = 320, 240

    m = load_mesh('data/nasa_earth.obj')

    # Create V, A, U, f: geometry, brightness, camera, renderer
    V = ch.array(m.v)
    A = SphericalHarmonics(vn=VertNormals(v=V, f=m.f),
                           components=[3., 2., 0., 0., 0., 0., 0., 0., 0.],
                           light_color=ch.ones(3))
    U = ProjectPoints(v=V, f=[w, w], c=[w / 2., h / 2.], k=ch.zeros(5),
                      t=ch.zeros(3), rt=ch.zeros(3))
    f = TexturedRenderer(vc=A, camera=U, f=m.f, bgcolor=[0., 0., 0.],
                         texture_image=m.texture_image, vt=m.vt, ft=m.ft,
                         frustum={'width': w, 'height': h, 'near': 1, 'far': 20})

    # Parameterize the vertices
    translation, rotation = ch.array([0, 0, 8]), ch.zeros(3)
    f.v = translation + V.dot(Rodrigues(rotation))

    observed = f.r
    np.random.seed(1)
    translation[:] = translation.r + np.random.rand(3)
    rotation[:] = rotation.r + np.random.rand(3) * .2
    A.components[1:] = 0

    # Create the energy
    E_raw = f - observed
    E_pyr = gaussian_pyramid(E_raw, n_levels=6, normalization='size')

    def cb(_):
        import cv2
        print("Inside callback")
        cv2.imshow('Absolute difference', np.abs(E_raw.r))

        cv2.waitKey(10)

    print('OPTIMIZING TRANSLATION, ROTATION, AND LIGHT PARMS')
    free_variables = [translation, rotation, A.components]
    print(A.components)
    ch.minimize({'pyr': E_pyr}, x0=free_variables, callback=cb)
    print("Start inner optimization")
    ch.minimize({'raw': E_raw}, x0=free_variables, callback=cb)
    print(A.components)

    cv2.waitKey(1)


if __name__ == '__main__':
    main()

