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


class MyFuncForward(Callback):
    def __init__(self, name, n_fwd, opts={}):
        Callback.__init__(self)
        self.n_fwd = n_fwd
        self.construct(name, opts)

    # Number of inputs and outputs
    def get_n_in(self):
        return 3

    def get_n_out(self):
        return 1

    def get_sparsity_in(self, idx):
        if idx == 0:
            return Sparsity.dense(2, 1)
        if idx == 1:
            return Sparsity.dense(4, 1)
        return Sparsity.dense(2, self.n_fwd)

    def get_sparsity_out(self, idx):
        return Sparsity.dense(4, self.n_fwd)

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

    def has_forward(self, nfwd):
        return True

    def get_forward(self, nfwd, name, inames, onames, opts):
        print(f"get_forward nfwd = {nfwd}, name = {name}, inames = {inames}, onames = {onames}")
        out = MyFuncForward(self.name() + f"fwd{nfwd}", nfwd)
        return out


def main():
    my_func = MyFunc("my_func")
    x = MX.sym("x", 2, 1)
    res = my_func(x)
    print(f"res = {res}")
    print(f"jac_regular = {casadi.jacobian(res, x)}")
    print(f"jac_jtimes = {casadi.jtimes(res, x, MX.eye(2))}")


if __name__ == '__main__':
    main()
