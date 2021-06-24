import jax
import jax.numpy as np
from jax.scipy.ndimage import map_coordinates


def fun1(image: np.ndarray, point: np.ndarray) -> np.ndarray:
    return map_coordinates(image, point.T, order=1)


def main():
    image = np.array([
        [0, 0, 13],
        [0, 100, 0],
        [0, 0, 0],
    ], dtype=np.float32)
    coords1 = np.array([[1, 1], [0, 2]], dtype=np.float32)

    jac1 = jax.jacrev(fun1)
    jac2 = jax.jacfwd(fun1)
    print("func val")
    print(fun1(image, coords1))
    print("jac val")
    print(jac2(image, coords1))


if __name__ == '__main__':
    main()
