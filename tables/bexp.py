from abc import ABC

from schema.helpers.find_index import find_index
from tables.exp import Exp


class Bexp(Exp, ABC):
    def __init__(self, code):
        self.code = code

    def __eq__(self, other):
        return EqualityBexp(self, other)

    def __and__(self, other):
        return AndBexp(self, other)

    def __rand__(self, other):
        return AndBexp(self, other)

    def __or__(self, other):
        return OrBexp(self, other)

    def __ror__(self, other):
        return OrBexp(self, other)

    def __invert__(self):
        return NotBexp(self)


class ColumnBexp(Bexp):
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
            return ColumnBexp(len(parameters)), parameters + [self.column.raw_column]
        else:
            return ColumnBexp(idx), parameters


class ConstBexp(Bexp):
    def __init__(self, constant):
        super().__init__("CNT")
        self.constant = constant

    def __repr__(self):
        return f"CONST <{self.constant}>"

    def __str__(self):
        return self.__repr__()

    def to_closure(self, parameters):
        return self, parameters


class EqualityBexp(Bexp):
    def __init__(self, lexp, rexp):
        super().__init__("EQ")
        self.lexp = lexp
        self.rexp = rexp

    def __repr__(self):
        return f"EQ <{self.lexp}, {self.rexp}>"

    def __str__(self):
        return self.__repr__()

    def to_closure(self, parameters):
        new_lexp, lparams = self.lexp.to_closure(parameters)
        new_rexp, rparams = self.rexp.to_closure(lparams)
        return EqualityBexp(new_lexp, new_rexp), rparams


class NABexp(Bexp):
    def __init__(self, exp):
        super().__init__("NA")
        self.exp = exp

    def __repr__(self):
        return f"NA <{self.exp}>"

    def __str__(self):
        return self.__repr__()

    def to_closure(self, parameters):
        new_exp, new_params = self.exp.to_closure(parameters)
        return NABexp(new_exp), new_params


class LessThanBexp(Bexp):
    def __init__(self, lexp, rexp):
        super().__init__("LT")
        self.lexp = lexp
        self.rexp = rexp

    def __repr__(self):
        return f"LT <{self.lexp}, {self.rexp}>"

    def __str__(self):
        return self.__repr__()

    def to_closure(self, parameters):
        new_lexp, lparams = self.lexp.to_closure(parameters)
        new_rexp, rparams = self.rexp.to_closure(lparams)
        return LessThanBexp(new_lexp, new_rexp), rparams


class AndBexp(Bexp):
    def __init__(self, lexp, rexp):
        super().__init__("AND")
        self.lexp = lexp
        self.rexp = rexp

    def __repr__(self):
        return f"AND <{self.lexp}, {self.rexp}>"

    def __str__(self):
        return self.__repr__()

    def to_closure(self, parameters):
        new_lexp, lparams = self.lexp.to_closure(parameters)
        new_rexp, rparams = self.rexp.to_closure(lparams)
        return AndBexp(new_lexp, new_rexp), rparams


class OrBexp(Bexp):
    def __init__(self, lexp, rexp):
        super().__init__("OR")
        self.lexp = lexp
        self.rexp = rexp

    def __repr__(self):
        return f"OR <{self.lexp}, {self.rexp}>"

    def __str__(self):
        return self.__repr__()

    def to_closure(self, parameters):
        new_lexp, lparams = self.lexp.to_closure(parameters)
        new_rexp, rparams = self.rexp.to_closure(lparams)
        return OrBexp(new_lexp, new_rexp), rparams


class NotBexp(Bexp):
    def __init__(self, exp):
        super().__init__("NOT")
        self.exp = exp

    def to_closure(self, parameters):
        subexp = self.exp
        new_exp, new_params = subexp.to_closure(parameters)
        return NotBexp(new_exp), new_params
