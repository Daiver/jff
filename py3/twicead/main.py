import math


class Variable:
    def __init__(self, value, der=0.0):
        self.value = value
        self.der = der

    def __add__(self, other):
        if isinstance(other, float):
            other = Variable(other, 0.0)
        return Variable(self.value + other.value, self.der + other.der)

    def __radd__(self, other):
        return self.__add__(other)

    def __sub__(self, other):
        if isinstance(other, float):
            other = Variable(other, 0.0)
        return Variable(self.value - other.value, self.der - other.der)

    def __mul__(self, other):
        if isinstance(other, float):
            other = Variable(other, 0.0)
        return Variable(self.value * other.value, self.der * other.value + other.der * self.value)

    def __rmul__(self, other):
        return self.__mul__(other)

    def __repr__(self):
        return f"(x={self.value} dx={self.der})"


def func(x):
    return x*x*x + 2.0 * x * x
    # return x * x * 12.0 + x * 5.0 + 6.0
    # return (13.0*x + 5.0) * (13.0*x + 5.0)


def func2(x, y):
    return 2.0 * x * x + 5.0 * y * y + 13.0 * x * y


def main():
    # x = Variable(5, 1)
    x = Variable(Variable(2.0, 0.0), Variable(1.0, 0.0))
    y = Variable(Variable(12.0, 1.0), Variable(0.0, 0.0))
    res = func2(x, y)
    print(res)


if __name__ == '__main__':
    main()

'''
class Variable
{
private:
float value;
float der;
}

template<typename T>
class Variable
{
private:
T value;
T der;
}

'''