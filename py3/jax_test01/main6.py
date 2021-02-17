import numpy as np
import jax
import jax.numpy as jnp
from jax.scipy.ndimage import map_coordinates


def remap(image: jnp.ndarray, point: jnp.ndarray) -> jnp.ndarray:
    return map_coordinates(image, point.T, order=1)


def mix_images(weight: jnp.ndarray) -> jnp.ndarray:
    image1 = jnp.array([
        [0, 0, 0],
        [0, 0, 0],
        [0, 0, 0],
    ], dtype=jnp.float32)
    image2 = jnp.array([
        [0, 0, 13],
        [0, 100, 0],
        [0, 0, 0],
    ], dtype=jnp.float32)
    return (1.0 - weight) * image1 + weight * image2


@jax.jit
def residuals_fn(weight: jnp.ndarray) -> jnp.ndarray:
    target_values = jnp.array([100, 13])
    coords = jnp.array([[1, 1], [0, 2]], dtype=jnp.float32)
    cur_img = mix_images(weight)
    current_values = remap(cur_img, coords)
    diff = current_values - target_values
    return diff


def main():
    jac_fn = jax.jit(jax.jacfwd(residuals_fn))

    weight = np.array([0], dtype=np.float32)
    for i in range(10):
        residuals_val = residuals_fn(weight)
        loss_val = residuals_val.T @ residuals_val
        jac_val = jac_fn(weight)
        grad_val = jac_val.T @ residuals_val
        print(f"{i} loss {loss_val} grad {grad_val} weight{weight}")
        solution = np.linalg.solve(jac_val.T @ jac_val, grad_val)
        weight -= solution


if __name__ == '__main__':
    main()
