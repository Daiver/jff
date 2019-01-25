
class Node:
    def __init__(self):
        pass

class Var(Node):
    def __init__(self, index):
        self.index = index
        self.value = None
        self.grad  = None

    def forward(self, vars):
        index = self.index
        self.value = vars[index]
        return self.value

    def backward(self, sensitivity):
        self.grad = sensitivity

    def collect(self, arr):
        arr[self.index] += self.grad

    def __str__(self):
        return '[X%s, %s, %s\']' % (str(self.index), str(self.value), str(self.grad))

class Scalar(Node):
    def __init__(self, value):
        self.value = value

    def forward(self, vars):
        return self.value

    def backward(self, sensitivity):
        pass

    def collect(self, arr):
        pass

    def __str__(self):
        return '[Const %s]' % (str(self.value))

class Add(Node):
    def __init__(self, node1, node2):
        self.node1 = node1
        self.node2 = node2
        self.value = None

    def forward(self, vars):
        self.value = self.node1.forward(vars) + self.node2.forward(vars)
        return self.value

    def backward(self, sensitivity):
        self.node1.backward(sensitivity)
        self.node2.backward(sensitivity)

    def collect(self, arr):
        self.node1.collect(arr)
        self.node2.collect(arr)

    def __str__(self):
        return '[%s + %s, %s]' % (self.node1.__str__(), self.node2.__str__(), str(self.value))

class Mul(Node):
    def __init__(self, node1, node2):
        self.node1 = node1
        self.node2 = node2
        self.value = None

    def forward(self, vars):
        self.value = self.node1.forward(vars) * self.node2.forward(vars)
        return self.value

    def backward(self, sensitivity):
        self.node1.backward(sensitivity * self.node2.value)
        self.node2.backward(sensitivity * self.node1.value)

    def collect(self, arr):
        self.node1.collect(arr)
        self.node2.collect(arr)

    def __str__(self):
        return '[%s * %s, %s]' % (self.node1.__str__(), self.node2.__str__(), str(self.value))

if __name__ == '__main__':
    graph = Mul(Var(0), Add(Var(1), Mul(Var(0), Scalar(2))))
    vars = [3, 4]
    graph.forward(vars)
    graph.backward(1)
    print graph
    res = [0, 0]
    graph.collect(res)
    print res
