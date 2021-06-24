import jax
import jax.numpy as np


def maker(indices: np.ndarray):
    def fun1(x: np.ndarray) -> np.ndarray:
        return x[indices]
    return fun1


def main():
    x = np.array([1, 2, 3], dtype=np.float32)
    # jac1 = jax.grad(fun1)
    # jac1 = jax.jacfwd(fun1)
    indices = np.array([
        [0, 1],
        [2, 1]
    ])
    fun1 = maker(indices)
    jac1 = jax.jacrev(fun1)
    print("func val")
    print(fun1(x))
    print("jac val")
    print(jac1(x))


if __name__ == '__main__':
    main()
