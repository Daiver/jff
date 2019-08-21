import numpy as np

from math import sin, cos

import casadi
from casadi import SX, MX, DM, Function, Callback, Sparsity


class MyCallback(Callback):
    def __init__(self, name, opts={}):
        Callback.__init__(self)
        self.construct(name, opts)

    def get_n_in(self): return 1

    def get_n_out(self): return 1

    def get_sparsity_in(self, *args):
        print(f"get_sparsity_in: {args}")
        return Sparsity.dense(2, 1)

    # Evaluate numerically
    def eval(self, arg):
        print(f"arg {arg}")
        x = casadi.sum1(arg[0])
        print(f"eval: x = {x}")
        f = sin(x)
        return [f]


def main():
    mc = MyCallback("MyCallback")
    x = MX.sym("x", 2, 1)
    res = mc(x)
    print(f"res = {res}")

    res_func = Function("func", [x], [res])
    res_value = res_func([np.pi, 2*np.pi])
    print(f"res([np.pi, 2*np.pi]) = {res_value}")


if __name__ == '__main__':
    main()
