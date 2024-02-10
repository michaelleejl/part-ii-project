import operator

from schema import Cardinality


class Function:
    def __init__(self, function, arguments, cardinality=Cardinality.MANY_TO_ONE):
        self.function = lambda args: function(*args)
        self.arguments = arguments
        self.cardinality = cardinality
        self.explicit_keys = []

    @classmethod
    def identity(cls, *args):
        return Function(lambda x: x, list(args), Cardinality.ONE_TO_ONE)

    @classmethod
    def combine(cls, functions, combinator, force_cardinality=None):
        args = [fn.arguments for fn in functions]
        cardinality = Cardinality.ONE_TO_ONE
        if force_cardinality is not None:
            cardinality = Cardinality.MANY_TO_ONE
        else:
            for f in functions:
                c = f.cardinality
                if c == Cardinality.MANY_TO_ONE:
                    cardinality = Cardinality.MANY_TO_ONE

        def f_combine_g(c):
            vals = []
            i = 0
            for fn in functions:
                f = fn.function
                a = len(fn.arguments)
                vals += [f(*c[i : i + a])]
                i += a
            return combinator(*vals)

        fun = Function(f_combine_g, args, cardinality)
        fun.function = f_combine_g
        return fun

    def combine_with(self, other, op):
        from tables.column import Column

        if isinstance(other, Function):
            return Function.combine([self, other], op)
        elif isinstance(other, Column):
            return Function.combine(
                [self, Function.identity(other)], op, Cardinality.MANY_TO_ONE
            )
        else:
            return Function.combine([self, Function.identity(other)], op)

    def __add__(self, other):
        return self.combine_with(other, operator.add)

    def __sub__(self, other):
        return self.combine_with(other, operator.sub)

    def __mul__(self, other):
        return self.combine_with(other, operator.mul)

    def __truediv__(self, other):
        return self.combine_with(other, operator.truediv)


def create_function(function):
    return lambda arguments: Function(
        function, list(arguments), Cardinality.MANY_TO_ONE
    )


def create_bijection(function):
    return lambda arguments: Function(function, list(arguments), Cardinality.ONE_TO_ONE)
