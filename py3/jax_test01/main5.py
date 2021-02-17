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


def loss_fn(weight: jnp.ndarray) -> jnp.ndarray:
    target_values = jnp.array([100, 13])
    coords = jnp.array([[1, 1], [0, 2]], dtype=jnp.float32)
    cur_img = mix_images(weight)
    current_values = remap(cur_img, coords)
    diff = current_values - target_values
    return diff.T @ diff


def main():
    grad_fn = jax.grad(loss_fn)

    weight = np.array([0], dtype=np.float32)
    lr = 0.00005
    for i in range(1000):
        loss_val = loss_fn(weight)
        grad_val = grad_fn(weight)
        print(f"{i} loss {loss_val} grad {grad_val} weight{weight}")
        weight += -lr * grad_val


if __name__ == '__main__':
    main()
