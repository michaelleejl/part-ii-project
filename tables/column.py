import operator

from tables.function import Function
from tables.predicate import EqualityPredicate, NotPredicate, LessThanPredicate, NAPredicate, OrPredicate, AndPredicate
from tables.raw_column import RawColumn


class Column:
    def __init__(self, raw_column: RawColumn):
        self.raw_column = raw_column

    def __eq__(self, other) -> EqualityPredicate:
        return EqualityPredicate(self.raw_column.name, other)

    def __bool__(self):
        raise NotImplemented()

    def __ne__(self, other) -> NotPredicate:
        return NotPredicate(EqualityPredicate(self.raw_column.name, other))

    def __lt__(self, other) -> LessThanPredicate:
        return LessThanPredicate(self.raw_column.name, other)

    def __gt__(self, other) -> NotPredicate:
        return NotPredicate(LessThanPredicate(self.raw_column.name, other))

    def __le__(self, other) -> OrPredicate:
        return OrPredicate(LessThanPredicate(self.raw_column.name, other), EqualityPredicate(self.raw_column.name, other))

    def __ge__(self, other) -> OrPredicate:
        return OrPredicate(NotPredicate(LessThanPredicate(self.raw_column.name, other)), EqualityPredicate(self.raw_column.name, other))

    def isnull(self) -> NAPredicate:
        return NAPredicate(self.raw_column.name)

    def isnotnull(self) -> NotPredicate:
        return NotPredicate(NAPredicate(self.raw_column.name))

    def __hash__(self):
        return self.raw_column.__hash__()

    def __add__(self, other):
        return Function(operator.add, [self, other])

    def __radd__(self, other):
        return self.__add__(other)

    def __sub__(self, other):
        pass

    def __mul__(self, other):
        pass

    def __truediv__(self, other):
        pass

    def get_explicit_keys(self):
        return self.raw_column.get_explicit_keys()

    def get_hidden_keys(self):
        return self.raw_column.get_hidden_keys()
