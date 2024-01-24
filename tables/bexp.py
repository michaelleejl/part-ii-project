from abc import ABC

from schema import BaseType
from schema.helpers.find_index import find_index
from tables.exp import Exp
from tables.helpers.wrap_bexp import wrap_bexp


class Bexp(Exp, ABC):
    def __init__(self, code):
        super().__init__(code, BaseType.BOOL)

    def __eq__(self, other):
        return EqualityBexp(self, wrap_bexp(other))

    def __and__(self, other):
        return AndBexp(self, wrap_bexp(other))

    def __rand__(self, other):
        return AndBexp(wrap_bexp(other), self)

    def __or__(self, other):
        return OrBexp(self, wrap_bexp(other))

    def __ror__(self, other):
        return OrBexp(wrap_bexp(other), self)

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

    def to_closure(self, parameters, aggregated_over):
        idx = find_index(self.column, parameters)
        if idx == -1:
            return ColumnBexp(len(parameters)), parameters + [self.column]
        else:
            return ColumnBexp(idx), parameters, aggregated_over


class ConstBexp(Bexp):
    def __init__(self, constant):
        super().__init__("CNT")
        self.constant = constant

    def __repr__(self):
        return f"CONST <{self.constant}>"

    def __str__(self):
        return self.__repr__()

    def to_closure(self, parameters, aggregated_over):
        return self, parameters, aggregated_over


class EqualityBexp(Bexp):
    def __init__(self, lexp, rexp):
        super().__init__("EQ")
        self.lexp = lexp
        self.rexp = rexp

    def __repr__(self):
        return f"EQ <{self.lexp}, {self.rexp}>"

    def __str__(self):
        return self.__repr__()

    def to_closure(self, parameters, aggregated_over):
        new_lexp, lparams, aggregated_over = self.lexp.to_closure(parameters, aggregated_over)
        new_rexp, rparams, aggregated_over = self.rexp.to_closure(lparams, aggregated_over)
        return EqualityBexp(new_lexp, new_rexp), rparams, aggregated_over


class NABexp(Bexp):
    def __init__(self, exp):
        super().__init__("NA")
        self.exp = exp

    def __repr__(self):
        return f"NA <{self.exp}>"

    def __str__(self):
        return self.__repr__()

    def to_closure(self, parameters, aggregated_over):
        new_exp, new_params, aggregated_over = self.exp.to_closure(parameters, aggregated_over)
        return NABexp(new_exp), new_params, aggregated_over


class LessThanBexp(Bexp):
    def __init__(self, lexp, rexp):
        super().__init__("LT")
        self.lexp = lexp
        self.rexp = rexp

    def __repr__(self):
        return f"LT <{self.lexp}, {self.rexp}>"

    def __str__(self):
        return self.__repr__()

    def to_closure(self, parameters, aggregated_over):
        new_lexp, lparams, aggregated_over = self.lexp.to_closure(parameters, aggregated_over)
        new_rexp, rparams, aggregated_over = self.rexp.to_closure(lparams, aggregated_over)
        return LessThanBexp(new_lexp, new_rexp), rparams, aggregated_over


class AndBexp(Bexp):
    def __init__(self, lexp, rexp):
        super().__init__("AND")
        self.lexp = lexp
        self.rexp = rexp

    def __repr__(self):
        return f"AND <{self.lexp}, {self.rexp}>"

    def __str__(self):
        return self.__repr__()

    def to_closure(self, parameters, aggregated_over):
        new_lexp, lparams, aggregated_over = self.lexp.to_closure(parameters, aggregated_over)
        new_rexp, rparams, aggregated_over = self.rexp.to_closure(lparams, aggregated_over)
        return AndBexp(new_lexp, new_rexp), rparams, aggregated_over


class OrBexp(Bexp):
    def __init__(self, lexp, rexp):
        super().__init__("OR")
        self.lexp = lexp
        self.rexp = rexp

    def __repr__(self):
        return f"OR <{self.lexp}, {self.rexp}>"

    def __str__(self):
        return self.__repr__()

    def to_closure(self, parameters, aggregated_over):
        new_lexp, lparams, aggregated_over = self.lexp.to_closure(parameters, aggregated_over)
        new_rexp, rparams, aggregated_over = self.rexp.to_closure(lparams, aggregated_over)
        return OrBexp(new_lexp, new_rexp), rparams, aggregated_over


class NotBexp(Bexp):
    def __init__(self, exp):
        super().__init__("NOT")
        self.exp = exp

    def to_closure(self, parameters, aggregated_over):
        subexp = self.exp
        new_exp, new_params, aggregated_over = subexp.to_closure(parameters, aggregated_over)
        return NotBexp(new_exp), new_params, aggregated_over


class AnyBexp(Bexp):

    def __init__(self, keys, column):
        super().__init__("ANY")
        self.keys = keys
        self.column = column

    def __repr__(self):
        return f"ANY <{self.keys}, {self.column}>"

    def to_closure(self, parameters, aggregated_over):
        key_idxs = [find_index(key, parameters) for key in self.keys]
        idx = find_index(self.column, parameters)
        aggregated_over = aggregated_over + [self.column]
        key_params, parameters = Exp.convert_agg_exp_variables(parameters, key_idxs, self.keys)
        if idx == -1:
            return AnyBexp(key_params, len(parameters)), parameters + [self.column], aggregated_over
        else:
            return AnyBexp(key_params, idx), parameters, aggregated_over


class AllBexp(Bexp):

    def __init__(self, keys, column):
        super().__init__("ALL")
        self.keys = keys
        self.column = column

    def __repr__(self):
        return f"ALL <{self.keys}, {self.column}>"

    def to_closure(self, parameters, aggregated_over):
        key_idxs = [find_index(key, parameters) for key in self.keys]
        idx = find_index(self.column.raw_column, parameters)
        aggregated_over = aggregated_over + [self.column]
        key_params, parameters = Exp.convert_agg_exp_variables(parameters, key_idxs, self.keys)
        if idx == -1:
            return AllBexp(key_params, len(parameters)), parameters + [self.column], aggregated_over
        else:
            return AllBexp(key_params, idx), parameters, aggregated_over