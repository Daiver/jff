import math
import taichi as ti


ti.init(default_fp=ti.f32)

dt = 0.001
learning_rate = 5
friction = 0.01

n_steps = 1024
max_steps = 1024

n_objects = 3
mass = 1
n_springs = 3
spring_stiffness = 10
damping = 20

x = ti.Vector(2, dt=ti.f32)
v = ti.Vector(2, dt=ti.f32)
force = ti.Vector(2, dt=ti.f32)
spring_length = ti.var(dt=ti.f32)
spring_anchor_a = ti.var(dt=ti.i32)
spring_anchor_b = ti.var(dt=ti.i32)
loss = ti.var(dt=ti.f32)


@ti.layout
def place():
    ti.root.dense(ti.l, max_steps).dense(ti.i, n_objects).place(x, v, force)
    ti.root.dense(ti.i, n_springs).place(spring_anchor_a, spring_anchor_b, spring_length)
    ti.root.place(loss)
    ti.root.lazy_grad()


@ti.kernel
def apply_spring_force(t: ti.i32):
    for i in range(n_springs):
        a, b = spring_anchor_a[i], spring_anchor_b[i]
        x_a, x_b = x[t - 1, a], x[t -1, b]
        diff = x_a - x_b
        length = diff.norm() + 1e-4
        F = (length - spring_length[i]) * spring_stiffness * diff / length
        ti.atomic_add(force[t, a], -F)
        ti.atomic_add(force[t, b], F)


@ti.kernel
def time_integrate(t: ti.i32):
    for i in range(n_objects):
        s = math.exp(-dt * damping)
        v[t, i] = s * v[t - 1, i] + dt * force[t, i] / mass
        x[t, i] = x[t - 1, i] + dt * v[t, i]


@ti.kernel
def compute_loss(t: ti.i32):
    x01 = x[t, 0] - x[t, 1]
    x02 = x[t, 0] - x[t, 2]
    area = ti.abs(
        0.5 * (x01[0] * x02[1] - x01[1] * x02[0])
    )
    target_area = 0.1
    loss[None] = ti.sqr(area - target_area)


def forward():
    for t in range(1, n_steps):
        apply_spring_force(t)
        time_integrate(t)
    compute_loss(n_steps - 1)


@ti.kernel
def clear_stages():
    for t in range(0, max_steps):
        for i in range(0, n_objects):
            x.grad[t, i] = ti.Vector([0.0, 0.0])
            v.grad[t, i] = ti.Vector([0.0, 0.0])
            force[t, i] = ti.Vector([0.0, 0.0])
            force.grad[t, i] = ti.Vector([0.0, 0.0])


@ti.kernel
def clear_springs():
    for i in range(n_springs):
        spring_length.grad[i] = 0.0


def clear_tensors():
    clear_stages()
    clear_springs()


def main():
    x[0, 0] = [0.3, 0.5]
    x[0, 1] = [0.3, 0.4]
    x[0, 2] = [0.4, 0.4]

    spring_anchor_a[0], spring_anchor_b[0], spring_length[0] = 0, 1, 0.1
    spring_anchor_a[1], spring_anchor_b[1], spring_length[1] = 1, 2, 0.1
    spring_anchor_a[2], spring_anchor_b[2], spring_length[2] = 2, 0, 0.1 * 2**0.5

    clear_tensors()
    forward()

    losses = []
    for iter in range(25):
        clear_tensors()
        with ti.Tape(loss):
            forward()
        print('Iter=', iter, 'Loss=', loss[None])
        losses.append(loss[None])

        for i in range(n_springs):
            spring_length[i] -= learning_rate * spring_length.grad[i]
        for i in range(n_springs):
            print(i, spring_length[i])


if __name__ == '__main__':
    main()
