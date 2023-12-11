import inspect
import operator


class Function:
    def __init__(self, function, arguments):
        self.function = lambda args: function(*args)
        self.arguments = arguments

    @classmethod
    def identity(cls, *args):
        return Function(lambda x: x, list(args))

    @classmethod
    def combine(cls, functions, combinator):
        args = [fn.arguments for fn in functions]

        def f_combine_g(c):
            vals = []
            i = 0
            for fn in functions:
                f = fn.function
                a = len(fn.arguments)
                vals += [f(*c[i:i + a])]
                i += a
            return combinator(*vals)

        fun = Function(f_combine_g, args)
        fun.function = f_combine_g
        return fun

    def __add__(self, other):
        if isinstance(other, Function):
            return Function.combine([self, other], operator.add)
        else:
            return Function.combine([self, Function.identity(other)], operator.add)


def create_function(function):
    return lambda arguments: Function(function, list(arguments))
