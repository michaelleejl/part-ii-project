from abc import ABC

from schema.helpers.find_index import find_index
from tables.exp import Exp


class Pexp(Exp, ABC):
    def __init__(self, code):
        self.code = code


class CountPexp(Pexp):

    def __init__(self, column):
        super().__init__("COU")
        self.column = column

    def __repr__(self):
        return f"COU <{self.column}>"

    def to_closure(self, parameters, aggregated_over):
        idx = find_index(self.column.raw_column, parameters)
        if idx == -1:
            return CountPexp(len(parameters)), parameters + [self.column.raw_column], aggregated_over
        else:
            return CountPexp(idx), parameters, aggregated_over


class MaxPexp(Pexp):

    def __init__(self, column):
        super().__init__("MAX")
        self.column = column

    def __repr__(self):
        return f"MAX <{self.column}>"

    def to_closure(self, parameters, aggregated_over):
        idx = find_index(self.column.raw_column, parameters)
        if idx == -1:
            return MaxPexp(len(parameters)), parameters + [self.column.raw_column], aggregated_over
        else:
            return MaxPexp(idx), parameters, aggregated_over


class PopPexp(Pexp):

    def __init__(self, column):
        super().__init__("POP")
        self.column = column

    def __repr__(self):
        return f"POP <{self.column}>"

    def to_closure(self, parameters, aggregated_over):
        idx = find_index(self.column.raw_column, parameters)
        if idx == -1:
            return PopPexp(len(parameters)), parameters + [self.column.raw_column], aggregated_over
        else:
            return PopPexp(idx), parameters, aggregated_over