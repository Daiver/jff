import numpy as np

from math import sin, cos

import casadi
from casadi import SX, MX, DM, Function, Callback, Sparsity


class MyCallback(Callback):
    def __init__(self, name, d, opts={}):
        Callback.__init__(self)
        self.d = d
        self.construct(name, opts)

    # Number of inputs and outputs
    def get_n_in(self): return 1

    def get_n_out(self): return 1

    def get_sparsity_in(self, *args) -> "casadi::Sparsity":
        """
        Get the sparsity of an input This function is called during construction.

        get_sparsity_in(self, int i) -> Sparsity





        """
        print(f"get_sparsity_in: {args}")
        return Sparsity.dense(2, 1)

    # def get_sparsity_out(self, *args) -> "casadi::Sparsity":
    #     """
    #     Get the sparsity of an output This function is called during construction.
    #
    #     get_sparsity_out(self, int i) -> Sparsity
    #
    #
    #
    #
    #
    #     """
    #     return _casadi.Callback_get_sparsity_out(self, *args)

    def has_jacobian(self, *args):
        return True

    def uses_output(self, *args):
        return False

    def get_jacobian(self, *args):
        """
        Return Jacobian of all input elements with respect to all output elements.

        get_jacobian(self, str name, [str] inames, [str] onames, dict opts) -> Function

        """
        jac_name, inames, onames, opts = args
        print("====")
        print("Inside get_jacobian")
        print(jac_name)
        print(inames)
        print("====")
        print(onames)
        print("====")
        print(opts)
        print("End of get_jacobian")
        print("====")

        i0 = MX.sym(inames[0], 1, 1)
        o0 = MX.sym(inames[1], 1, 1)
        # return Function(f"{jac_name}", [i0], [casadi.cos(i0 * self.d) * self.d])
        return Function(f"{jac_name}", [i0, o0], [casadi.cos(i0 * self.d) * self.d])

    # Initialize the object
    def init(self):
        print('initializing object')

    # Evaluate numerically
    def eval(self, arg):
        print(f"arg {arg}")
        x = arg[0][0]
        print(f"eval: x = {x}")
        f = sin(self.d*x)
        return [f]


def main():
    x = MX.sym("x", 2, 1)
    mc = MyCallback("MyCallback", 1)
    res = mc(x)
    print(res)
    # jac = casadi.jacobian(res, x)
    # print(jac)

    res_func = Function("func", [x], [res])
    # jac_func = Function("jac_func", [x], [jac])
    print(res_func([np.pi, 2*np.pi]))
    # print(jac_func([np.pi, np.pi]))


if __name__ == '__main__':
    main()
