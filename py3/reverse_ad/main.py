import numpy as np


class Node:
    def __init__(self, requires_grad=False):
        self.requires_grad = requires_grad

    def forward(self):
        raise NotImplementedError()

    def backward(self, sensitivity):
        raise NotImplementedError()

    def zero_grad(self):
        pass

    def __repr__(self):
        raise NotImplementedError()

    def __add__(self, other):
        return AddNode(self, other)
    
    def __mul__(self, other):
        return MulNode(self, other)
    
    def __matmul__(self, other):
        return MatMulNode(self, other)
    

class VarNode(Node):
    def __init__(self, value, requires_grad=False):
        super().__init__(requires_grad)
        self.value = value
        if self.requires_grad:
            self.grad = 0
        else:
            self.grad = None

    def backward(self, sensitivity):
        self.grad += sensitivity

    def zero_grad(self):
        if self.requires_grad:
            self.grad = 0

    def __repr__(self):
        return str(self.value)

class BinNode(Node):
    def __init__(self, lhs, rhs):
        super().__init__(requires_grad=lhs.requires_grad or rhs.requires_grad)
        self.lhs = lhs
        self.rhs = rhs

    def zero_grad(self):
        self.lhs.zero_grad()
        self.rhs.zero_grad()


class AddNode(BinNode):
    def __init__(self, lhs, rhs):
        super().__init__(lhs, rhs)
        self.value = lhs.value + rhs.value

    def backward(self, sensitivity):
        if self.lhs.requires_grad:
            self.lhs.backward(sensitivity)
        if self.rhs.requires_grad:
            self.rhs.backward(sensitivity)
    
    def __repr__(self):
        return str(self.lhs) + ' + ' + str(self.rhs)


class MulNode(BinNode):
    def __init__(self, lhs, rhs):
        super().__init__(lhs, rhs)
        self.value = lhs.value * rhs.value

    def backward(self, sensitivity):
        if self.lhs.requires_grad:
            self.lhs.backward(sensitivity * self.rhs.value)
        if self.rhs.requires_grad:
            self.rhs.backward(sensitivity * self.lhs.value)
    
    def __repr__(self):
        return str(self.lhs) + ' * ' + str(self.rhs)


class MatMulNode(BinNode):
    def __init__(self, lhs, rhs):
        super().__init__(lhs, rhs)
        self.value = lhs.value * rhs.value

    def backward(self, sensitivity):
        if self.lhs.requires_grad:
            self.lhs.backward(sensitivity * self.rhs.value)
        if self.rhs.requires_grad:
            self.rhs.backward(sensitivity * self.lhs.value)
    
    def __repr__(self):
        return str(self.lhs) + ' @ ' + str(self.rhs)


if __name__ == '__main__':
    a = VarNode(np.array([2]), True)
    b = VarNode(np.array([3]), False)
    c = VarNode(np.array([5]), True)

    #res = a + b
    #res = a + b * c
    res = a + b @ c
    res.backward(1.0)
    print(res)
    print(res.value)
    print(a.grad)
    print(b.grad)
    print(c.grad)



