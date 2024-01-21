import abc

from schema import BaseType
from schema.helpers.find_index import find_index
from tables.exp import Exp
from tables.helpers.wrap_aexp import wrap_aexp


class Aexp(Exp, abc.ABC):
    def __init__(self, code):
        super().__init__(code, BaseType.FLOAT)

    def __add__(self, other):
        return AddAexp(self, wrap_aexp(other))

    def __radd__(self, other):
        return wrap_aexp(other).__add__(self)

    def __sub__(self, other):
        return SubAexp(self, wrap_aexp(other))

    def __rsub__(self, other):
        return wrap_aexp(other).__sub__(self)

    def __mul__(self, other):
        return MulAexp(self, wrap_aexp(other))

    def __rmul__(self, other):
        return wrap_aexp(other).__mul__(self)

    def __truediv__(self, other):
        return DivAexp(self, wrap_aexp(other))

    def __rtruediv__(self, other):
        return wrap_aexp(other).__rtruediv__(self)


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

    def to_closure(self, parameters, aggregated_over):
        idx = find_index(self.column, parameters)
        if idx == -1:
            return ColumnAexp(len(parameters)), parameters + [self.column], aggregated_over
        else:
            return ColumnAexp(idx), parameters, aggregated_over


class ConstAexp(Aexp):
    def __init__(self, constant):
        super().__init__("CNT")
        self.constant = constant

    def __repr__(self):
        return f"CONST <{self.constant}>"

    def __str__(self):
        return self.__repr__()

    def to_closure(self, parameters, aggregated_over):
        return self, parameters, aggregated_over


class AddAexp(Aexp):
    def __init__(self, lexp, rexp):
        super().__init__("ADD")
        self.lexp = lexp
        self.rexp = rexp

    def __repr__(self):
        return f"ADD <{self.lexp}, {self.rexp}>"

    def __str__(self):
        return self.__repr__()

    def to_closure(self, parameters, aggregated_over):
        new_lexp, lparams, aggregated_over = self.lexp.to_closure(parameters, aggregated_over)
        new_rexp, rparams, aggregated_over = self.rexp.to_closure(lparams, aggregated_over)
        return AddAexp(new_lexp, new_rexp), rparams, aggregated_over


class SubAexp(Aexp):
    def __init__(self, lexp, rexp):
        super().__init__("SUB")
        self.lexp = lexp
        self.rexp = rexp

    def __repr__(self):
        return f"SUB <{self.lexp}, {self.rexp}>"

    def __str__(self):
        return self.__repr__()

    def to_closure(self, parameters, aggregated_over):
        new_lexp, lparams, aggregated_over = self.lexp.to_closure(parameters, aggregated_over)
        new_rexp, rparams, aggregated_over = self.rexp.to_closure(lparams, aggregated_over)
        return SubAexp(new_lexp, new_rexp), rparams, aggregated_over


class NegAexp(Aexp):
    def __init__(self, exp):
        super().__init__("NEG")
        self.exp = exp

    def __repr__(self):
        return f"NEG <{self.exp}>"

    def __str__(self):
        return self.__repr__()

    def to_closure(self, parameters, aggregated_over):
        subexp = self.exp
        new_exp, new_params, aggregated_over = subexp.to_closure(parameters, aggregated_over)
        return NegAexp(new_exp), new_params, aggregated_over


class MulAexp(Aexp):

    def __init__(self, lexp, rexp):
        super().__init__("MUL")
        self.lexp = lexp
        self.rexp = rexp

    def __repr__(self):
        return f"MUL <{self.lexp}, {self.rexp}>"

    def __str__(self):
        return self.__repr__()

    def to_closure(self, parameters, aggregated_over):
        new_lexp, lparams, aggregated_over = self.lexp.to_closure(parameters, aggregated_over)
        new_rexp, rparams, aggregated_over = self.rexp.to_closure(lparams, aggregated_over)
        return MulAexp(new_lexp, new_rexp), rparams, aggregated_over


class DivAexp(Aexp):

    def __init__(self, lexp, rexp):
        super().__init__("DIV")
        self.lexp = lexp
        self.rexp = rexp

    def __repr__(self):
        return f"DIV <{self.lexp}, {self.rexp}>"

    def __str__(self):
        return self.__repr__()

    def to_closure(self, parameters, aggregated_over):
        new_lexp, lparams, aggregated_over = self.lexp.to_closure(parameters, aggregated_over)
        new_rexp, rparams, aggregated_over = self.rexp.to_closure(lparams, aggregated_over)
        return DivAexp(new_lexp, new_rexp), rparams, aggregated_over


class SumAexp(Aexp):

    def __init__(self, keys, column):
        super().__init__("SUM")
        self.keys = keys
        self.column = column

    def __repr__(self):
        return f"SUM <{self.column}>"

    def to_closure(self, parameters, aggregated_over):
        key_idxs = [find_index(key, parameters) for key in self.keys]
        idx = find_index(self.column.raw_column, parameters)
        aggregated_over = aggregated_over + [self.column.raw_column]
        key_params, parameters = Exp.convert_agg_exp_variables(parameters, key_idxs, self.keys)
        if idx == -1:
            return SumAexp(key_params, len(parameters)), parameters + [self.column.raw_column], aggregated_over
        else:
            return SumAexp(key_params, idx), parameters, aggregated_over


class MaxAexp(Aexp):

    def __init__(self, keys, column):
        super().__init__("MAX")
        self.keys = keys
        self.column = column

    def __repr__(self):
        return f"MAX <{self.keys}, {self.column}>"

    def to_closure(self, parameters, aggregated_over):
        key_idxs = [find_index(key, parameters) for key in self.keys]
        idx = find_index(self.column.raw_column, parameters)
        aggregated_over = aggregated_over + [self.column.raw_column]
        key_params, parameters = Exp.convert_agg_exp_variables(parameters, key_idxs, self.keys)
        if idx == -1:
            return MaxAexp(key_params, len(parameters)), parameters + [self.column.raw_column], aggregated_over
        else:
            return MaxAexp(key_params, idx), parameters, aggregated_over


class MinAexp(Aexp):

    def __init__(self, keys, column):
        super().__init__("MIN")
        self.keys = keys
        self.column = column

    def __repr__(self):
        return f"MIN <{self.keys}, {self.column}>"

    def to_closure(self, parameters, aggregated_over):
        key_idxs = [find_index(key, parameters) for key in self.keys]
        idx = find_index(self.column.raw_column, parameters)
        aggregated_over = aggregated_over + [self.column.raw_column]
        key_params, parameters = Exp.convert_agg_exp_variables(parameters, key_idxs, self.keys)
        if idx == -1:
            return MinAexp(key_params, len(parameters)), parameters + [self.column.raw_column], aggregated_over
        else:
            return MinAexp(key_params, idx), parameters, aggregated_over

class CountAexp(Aexp):

    def __init__(self, keys, column):
        super().__init__("COU")
        self.keys = keys
        self.column = column

    def __repr__(self):
        return f"COU <{self.keys}, {self.column}>"

    def to_closure(self, parameters, aggregated_over):
        key_idxs = [find_index(key, parameters) for key in self.keys]
        idx = find_index(self.column.raw_column, parameters)
        aggregated_over = aggregated_over + [self.column.raw_column]
        key_params, parameters = Exp.convert_agg_exp_variables(parameters, key_idxs, self.keys)
        if idx == -1:
            return CountAexp(key_params, len(parameters)), parameters + [self.column.raw_column], aggregated_over
        else:
            return CountAexp(key_params, idx), parameters, aggregated_over