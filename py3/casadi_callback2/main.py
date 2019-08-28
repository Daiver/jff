import numpy as np
import casadi
from casadi import MX, DM, Function, Callback, Sparsity


class MyCallbackJacobian(Callback):
    def __init__(self, name, n_items, opts={}):
        Callback.__init__(self)
        self.n_items = n_items
        self.construct(name, opts)

    # Number of inputs and outputs
    def get_n_in(self): return 2

    def get_n_out(self): return 1

    def get_sparsity_in(self, *args) -> "casadi::Sparsity":
        if args[0] == 0:
            return Sparsity.dense(self.n_items)
        return Sparsity.dense(self.n_items)

    def get_sparsity_out(self, *args) -> "casadi::Sparsity":
        return Sparsity.dense(self.n_items, self.n_items)

    def has_jacobian(self, *args):
        return False

    def uses_output(self, *args):
        return False

    def eval(self, arg):
        return [np.eye(self.n_items)]


class MyCallback(Callback):
    def __init__(self, name, n_items, opts={}):
        Callback.__init__(self)
        self.n_items = n_items
        self.jacobian_func = MyCallbackJacobian("jac_" + name, self.n_items)
        self.construct(name, opts)

    # Number of inputs and outputs
    def get_n_in(self): return 1

    def get_n_out(self): return 1

    def get_sparsity_in(self, *args) -> "casadi::Sparsity":
        return Sparsity.dense(self.n_items)

    def get_sparsity_out(self, *args) -> "casadi::Sparsity":
        return Sparsity.dense(self.n_items)

    def has_jacobian(self, *args):
        return True

    def uses_output(self, *args):
        return False

    def get_jacobian(self, *args):
        return self.jacobian_func

    def eval(self, arg):
        x = arg[0]
        return [x]


def main():
    print(f"Casadi version {casadi.__version__}")
    n_items = 300
    x = MX.sym("x", 1)
    # x = MX.zeros(n_items)
    mc = MyCallback("MyCallback", n_items)
    points = MX.zeros(n_items) + x
    res = mc(points)
    print(f"res = {res}")
    jac = casadi.jacobian(res, x, {})
    print(f"jac = {jac}")
    jac_func = Function("tmp", [x], [jac])
    print(f"jac_func(value) = {jac_func(np.zeros(x.shape[0]))}")


if __name__ == '__main__':
    main()
