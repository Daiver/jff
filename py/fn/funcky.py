import operator

class AnonHelper:
    def evalOrAsIs(self, arg, args):
        return arg.evaluate(args) if isinstance(arg, AnonHelper) else arg

    def __add__(self, arg):
        return AnonOperator(operator.add, self, arg)

    def __radd__(self, arg):
        return AnonOperator(operator.add, arg, self)

    def __sub__(self, arg):
        return AnonOperator(operator.sub, self, arg)

    def __rsub__(self, arg):
        return AnonOperator(operator.sub, arg, self)

    def __mul__(self, arg):
        return AnonOperator(operator.mul, self, arg)

    def __rmul__(self, arg):
        return AnonOperator(operator.mul, arg, self)

    def __div__(self, arg):
        return AnonOperator(operator.div, self, arg)

    def __rdiv__(self, arg):
        return AnonOperator(operator.div, arg, self)

    def __rmod__(self, arg):
        return AnonOperator(operator.mod, arg, self)

    def __mod__(self, arg):
        return AnonOperator(operator.mod, self, arg)

    def __gt__(self, arg):
        return AnonOperator(operator.gt, self, arg)

    def __ge__(self, arg):
        return AnonOperator(operator.ge, self, arg)

    def __lt__(self, arg):
        return AnonOperator(operator.lt, self, arg)

    def __le__(self, arg):
        return AnonOperator(operator.le, self, arg)

    def __eq__(self, arg):
        return AnonOperator(operator.eq, self, arg)

    def __ne__(self, arg):
        return AnonOperator(operator.ne, self, arg)

    def __call__(self, *args):
        return self.evaluate(args)


class ArgumentPlaceholder(AnonHelper) :
    def __init__(self, place):
        self.place = place

    def evaluate(self, args):
        if len(args) <= self.place:
            return ArgumentPlaceholder(self.place - len(args))
        return args[self.place]

class AnonOperator(AnonHelper):
    def __init__(self, op, arg1, arg2):
        self.arg1 = arg1
        self.arg2 = arg2
        self.op = op

    def evaluate(self, args):
        arg1 = self.evalOrAsIs(self.arg1, args)
        arg2 = self.evalOrAsIs(self.arg2, args)
        return self.op(arg1, arg2)



_  = ArgumentPlaceholder(0)
_1 = ArgumentPlaceholder(0)
_2 = ArgumentPlaceholder(1)
_3 = ArgumentPlaceholder(2)
_4 = ArgumentPlaceholder(3)
_5 = ArgumentPlaceholder(4)
