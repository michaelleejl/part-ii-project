from abc import ABC

from schema import BaseType
from schema.helpers.find_index import find_index
from tables.exp import Exp


class Sexp(Exp, ABC):
    def __init__(self, code):
        super().__init__(code, BaseType.STRING)


class ColumnSexp(Sexp):
    def __init__(self, column):
        super().__init__("COL")
        self.column = column

    def __repr__(self):
        return f"COL <{self.column}>"

    def __str__(self):
        return self.__repr__()

    def to_closure(self, parameters, aggregated_over):
        idx = find_index(self.column.raw_column, parameters)
        if idx == -1:
            return ColumnSexp(len(parameters)), parameters + [self.column.raw_column], aggregated_over
        else:
            return ColumnSexp(idx), parameters, aggregated_over


class ConstSexp(Sexp):
    def __init__(self, constant):
        super().__init__("CNT")
        self.constant = constant

    def __repr__(self):
        return f"CNT <{self.constant}>"

    def __str__(self):
        return self.__repr__()

    def to_closure(self, parameters, aggregated_over):
        return self, parameters, aggregated_over