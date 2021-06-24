import jax
import jax.numpy as np


def fun1(x: np.ndarray, indices: np.ndarray) -> np.ndarray:
    return x[indices]


def main():
    x = np.array([1, 2, 3], dtype=np.float32)
    # jac1 = jax.grad(fun1)
    # jac1 = jax.jacfwd(fun1)
    indices = np.array([
        0, 1,
    ])
    jac1 = jax.jacrev(fun1)
    print("func val")
    print(fun1(x, indices))
    print("jac val")
    print(jac1(x, indices))


if __name__ == '__main__':
    main()
