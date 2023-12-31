from abc import ABC

from schema.helpers.find_index import find_index
from tables.exp import Exp


class Fexp(Exp, ABC):
    def __init__(self, code):
        self.code = code


class SumFexp(Fexp):

    def __init__(self, column):
        super().__init__("SUM")
        self.column = column

    def __repr__(self):
        return f"SUM <{self.column}>"

    def to_closure(self, parameters):
        idx = find_index(self.column.raw_column, parameters)
        if idx == -1:
            return SumFexp(len(parameters)), parameters + [self.column.raw_column]
        else:
            return SumFexp(idx), parameters


class CountFexp(Fexp):

    def __init__(self, column):
        super().__init__("COU")
        self.column = column

    def __repr__(self):
        return f"COU <{self.column}>"

    def to_closure(self, parameters):
        idx = find_index(self.column.raw_column, parameters)
        if idx == -1:
            return CountFexp(len(parameters)), parameters + [self.column.raw_column]
        else:
            return CountFexp(idx), parameters


class MaxFexp(Fexp):

    def __init__(self, column):
        super().__init__("MAX")
        self.column = column

    def __repr__(self):
        return f"MAX <{self.column}>"

    def to_closure(self, parameters):
        idx = find_index(self.column.raw_column, parameters)
        if idx == -1:
            return MaxFexp(len(parameters)), parameters + [self.column.raw_column]
        else:
            return MaxFexp(idx), parameters


class PopFexp(Fexp):

    def __init__(self, column):
        super().__init__("POP")
        self.column = column

    def __repr__(self):
        return f"POP <{self.column}>"

    def to_closure(self, parameters):
        idx = find_index(self.column.raw_column, parameters)
        if idx == -1:
            return PopFexp(len(parameters)), parameters + [self.column.raw_column]
        else:
            return PopFexp(idx), parameters


class AnyFexp(Fexp):

    def __init__(self, column):
        super().__init__("ANY")
        self.column = column

    def __repr__(self):
        return f"ANY <{self.column}>"

    def to_closure(self, parameters):
        idx = find_index(self.column.raw_column, parameters)
        if idx == -1:
            return AnyFexp(len(parameters)), parameters + [self.column.raw_column]
        else:
            return AnyFexp(idx), parameters