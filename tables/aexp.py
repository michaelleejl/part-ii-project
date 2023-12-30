import abc

from schema.helpers.find_index import find_index
from tables.exp import Exp


class Aexp(Exp, abc.ABC):
    def __init__(self, code):
        self.code = code

    def __add__(self, other):
        return AddAexp(self, other)

    def __sub__(self, other):
        return SubAexp(self, other)

    def __mul__(self, other):
        return MulAexp(self, other)

    def __truediv__(self, other):
        return DivAexp(self, other)

    def __neg__(self):
        return NegAexp(self)


class ColumnAexp(Aexp):
    def __init__(self, column):
        super().__init__("COL")
        self.column = column

    def __repr__(self):
        return f"COL <{self.column}>"

    def __str__(self):
        return self.__repr__()

    def to_closure(self, parameters):
        idx = find_index(self.column.raw_column, parameters)
        if idx == -1:
            return ColumnAexp(len(parameters)), parameters + [self.column.raw_column]
        else:
            return ColumnAexp(idx), parameters


class ConstAexp(Aexp):
    def __init__(self, constant):
        super().__init__("CNT")
        self.constant = constant

    def __repr__(self):
        return f"CONST <{self.constant}>"

    def __str__(self):
        return self.__repr__()

    def to_closure(self, parameters):
        return self, parameters


class AddAexp(Aexp):
    def __init__(self, lexp, rexp):
        super().__init__("ADD")
        self.lexp = lexp
        self.rexp = rexp

    def __repr__(self):
        return f"ADD <{self.lexp}, {self.rexp}>"

    def __str__(self):
        return self.__repr__()

    def to_closure(self, parameters):
        new_lexp, lparams = self.lexp.to_closure(parameters)
        new_rexp, rparams = self.rexp.to_closure(lparams)
        return AddAexp(new_lexp, new_rexp), rparams


class SubAexp(Aexp):
    def __init__(self, lexp, rexp):
        super().__init__("SUB")
        self.lexp = lexp
        self.rexp = rexp

    def __repr__(self):
        return f"SUB <{self.lexp}, {self.rexp}>"

    def __str__(self):
        return self.__repr__()

    def to_closure(self, parameters):
        new_lexp, lparams = self.lexp.to_closure(parameters)
        new_rexp, rparams = self.rexp.to_closure(lparams)
        return SubAexp(new_lexp, new_rexp), rparams


class NegAexp(Aexp):
    def __init__(self, exp):
        super().__init__("NEG")
        self.exp = exp

    def __repr__(self):
        return f"NEG <{self.exp}>"

    def __str__(self):
        return self.__repr__()

    def to_closure(self, parameters):
        subexp = self.exp
        new_exp, new_params = subexp.to_closure(parameters)
        return NegAexp(new_exp), new_params


class MulAexp(Aexp):

    def __init__(self, lexp, rexp):
        super().__init__("MUL")
        self.lexp = lexp
        self.rexp = rexp

    def __repr__(self):
        return f"MUL <{self.lexp}, {self.rexp}>"

    def __str__(self):
        return self.__repr__()

    def to_closure(self, parameters):
        new_lexp, lparams = self.lexp.to_closure(parameters)
        new_rexp, rparams = self.rexp.to_closure(lparams)
        return MulAexp(new_lexp, new_rexp), rparams


class DivAexp(Aexp):

    def __init__(self, lexp, rexp):
        super().__init__("DIV")
        self.lexp = lexp
        self.rexp = rexp

    def __repr__(self):
        return f"DIV <{self.lexp}, {self.rexp}>"

    def __str__(self):
        return self.__repr__()

    def to_closure(self, parameters):
        new_lexp, lparams = self.lexp.to_closure(parameters)
        new_rexp, rparams = self.rexp.to_closure(lparams)
        return DivAexp(new_lexp, new_rexp), rparams
