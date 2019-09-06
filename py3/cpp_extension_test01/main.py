import casadi
from casadi import Callback
import demo


class A:
    def __int__(self):
        return 123


print(demo.add(12, 3))
print(demo.foo(A()))
