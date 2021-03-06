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
        # return Sparsity.dense(self.n_items, self.n_items)
        return Sparsity.diag(self.n_items)

    def has_jacobian(self, *args):
        return False

    def uses_output(self, *args):
        return False

    def eval(self, arg):
        return [DM(self.get_sparsity_out(0), self.n_items * [1])]


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
    n_items = 3
    mc = MyCallback("MyCallback", n_items)
    # x = MX.sym("x", 1)
    x = MX.sym("x", n_items)
    # points = MX.zeros(n_items) + x
    points = x
    res = mc(points)
    print(f"res = {res}")
    # jac = casadi.jacobian(res, x, {})
    # v = MX.sym("v", x.shape[0])
    jac = casadi.jtimes(res, x, MX.eye(x.shape[0]))
    print(f"jac = {jac}")
    jac_func = Function("tmp", [x], [jac])
    print(f"jac_func(value) = {jac_func(DM(np.zeros(x.shape[0])))}")
    print(jac.sparsity())


if __name__ == '__main__':
    main()
