import jax
import jax.numpy as np


def fun1(x: np.ndarray):
    return x*x


def main():
    x = np.array([1, 2, 3], dtype=np.float32)
    # jac1 = jax.grad(fun1)
    jac1 = jax.jacfwd(fun1)
    print(fun1(x))
    print(jac1(x))


if __name__ == '__main__':
    main()
