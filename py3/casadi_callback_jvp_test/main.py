import casadi
from casadi import MX, Function, Callback, Sparsity


class MyFuncJacobian(Callback):
    def __init__(self, name, opts={}):
        Callback.__init__(self)

        self.construct(name, opts)

    # Number of inputs and outputs
    def get_n_in(self):
        return 2

    def get_n_out(self):
        return 1

    def get_sparsity_in(self, idx):
        if idx == 0:
            return Sparsity.dense(2, 1)
        return Sparsity.dense(4, 1)

    def get_sparsity_out(self, idx):
        return Sparsity.dense(4, 2)

    # Initialize the object
    def init(self):
        print('initializing object')

    # Evaluate numerically
    def eval(self, arg):
        x = arg[0]
        f = sin(self.d*x)
        return [f]


class MyFunc(Callback):
    def __init__(self, name, opts={}):
        Callback.__init__(self)
        self.jac_callback = MyFuncJacobian(name + "_jac")
        self.construct(name, opts)

    # Number of inputs and outputs
    def get_n_in(self):
        return 1

    def get_n_out(self):
        return 1

    def get_sparsity_in(self, idx):
        return Sparsity.dense(2, 1)

    def get_sparsity_out(self, idx):
        return Sparsity.dense(4, 1)

    # Initialize the object
    def init(self):
        print('initializing object')

    # Evaluate numerically
    def eval(self, arg):
        x = arg[0]
        f = sin(self.d*x)
        return [f]

    def has_jacobian(self):
        return True

    def get_jacobian(self, *args):
        return self.jac_callback


def main():
    my_func = MyFunc("my_func")
    x = MX.sym("x", 2, 1)
    res = my_func(x)
    print(f"res = {res}")
    print(f"jac = {casadi.jacobian(res, x)}")
    print(f"jac = {casadi.jtimes(res, x, MX.sym('v', 2, 1))}")
    j = casadi.jacobian(res, x)
    print(f"jac = {j.T@j}")
    casadi.DM()


if __name__ == '__main__':
    main()
