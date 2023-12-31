from schema import Cardinality, BaseType
from tables.aexp import ConstAexp, ColumnAexp, Aexp, AddAexp, SubAexp, MulAexp, DivAexp, SumAexp, MaxAexp, NegAexp, \
    CountAexp, MinAexp
from tables.aggregation import AggregationFunction
from tables.exceptions import ColumnTypeException
from tables.exp import PopExp
from tables.function import Function, create_function, create_bijection
from tables.bexp import EqualityBexp, NotBexp, LessThanBexp, NABexp, OrBexp, AndBexp, ColumnBexp, ConstBexp, Bexp, \
    AnyBexp, AllBexp
from tables.helpers.wrap_aexp import wrap_aexp
from tables.helpers.wrap_bexp import wrap_bexp
from tables.helpers.wrap_sexp import wrap_sexp
from tables.raw_column import RawColumn


def get_arguments_for_binary_aexp(x, y):
    lexp = wrap_aexp(x)
    rexp = wrap_aexp(y)
    if lexp is None or rexp is None:
        raise ColumnTypeException("float", "other")
    return lexp, rexp


def get_arguments_for_binary_bexp(x, y):
    lexp = wrap_bexp(x)
    rexp = wrap_bexp(y)
    if lexp is None or rexp is None:
        raise ColumnTypeException("bool", "other")
    return lexp, rexp


class Column:
    def __init__(self, raw_column: RawColumn):
        self.raw_column = raw_column

    # Boolean expressions

    def __eq__(self, other) -> EqualityBexp:
        data_type = self.raw_column.node.node_type
        lexp = None
        rexp = None
        if data_type == BaseType.BOOL:
            lexp = wrap_bexp(self)
            rexp = wrap_bexp(other)
        elif data_type == BaseType.FLOAT:
            lexp = wrap_aexp(self)
            rexp = wrap_aexp(other)
        elif data_type == BaseType.STRING:
            lexp = wrap_sexp(self)
            rexp = wrap_sexp(other)
        if lexp is None or rexp is None:
            raise ColumnTypeException(str(data_type), "other")
        return EqualityBexp(lexp, rexp)

    def __ne__(self, other) -> NotBexp:
        return NotBexp(self.raw_column.name == other)

    def __lt__(self, other) -> LessThanBexp:
        data_type = self.raw_column.node.node_type
        if data_type != BaseType.FLOAT:
            raise ColumnTypeException("float", str(data_type))
        lexp, rexp = get_arguments_for_binary_aexp(self, other)
        return LessThanBexp(lexp, rexp)

    def __gt__(self, other) -> AndBexp:
        return AndBexp(NotBexp(self < other), self != other)

    def __le__(self, other) -> OrBexp:
        return OrBexp(self < other, self == other)

    def __ge__(self, other) -> NotBexp:
        return NotBexp(self < other)

    def __and__(self, other):
        data_type = self.raw_column.node.node_type
        if data_type != BaseType.BOOL:
            raise ColumnTypeException("bool", str(data_type))
        lexp, rexp = get_arguments_for_binary_bexp(self, other)
        return AndBexp(lexp, rexp)

    def __rand__(self, other):
        self.__and__(other)

    def __or__(self, other):
        data_type = self.raw_column.node.node_type
        if data_type != BaseType.BOOL:
            raise ColumnTypeException("bool", str(data_type))
        lexp, rexp = get_arguments_for_binary_bexp(self, other)
        return OrBexp(lexp, rexp)

    def __ror__(self, other):
        self.__or__(other)

    def __invert__(self):
        data_type = self.raw_column.node.node_type
        if data_type != BaseType.BOOL:
            raise ColumnTypeException("bool", str(data_type))
        exp = wrap_bexp(self)
        return NotBexp(exp)

    def any(self):
        data_type = self.raw_column.node.node_type
        if data_type != BaseType.BOOL:
            raise ColumnTypeException("bool", str(data_type))
        keys = self.raw_column.get_strong_keys()
        return AnyBexp(keys, self)

    def all(self):
        data_type = self.raw_column.node.node_type
        if data_type != BaseType.BOOL:
            raise ColumnTypeException("bool", str(data_type))
        keys = self.raw_column.get_strong_keys()
        return AllBexp(keys, self)

    def isnull(self) -> NABexp:
        data_type = self.raw_column.node.node_type
        if data_type == BaseType.FLOAT:
            exp = ColumnAexp(self)
        elif data_type == BaseType.BOOL:
            exp = ColumnBexp(self)
        else:
            raise NotImplemented()
        return NABexp(exp)

    def isnotnull(self) -> NotBexp:
        return NotBexp(self.isnull())

    def to_bexp(self):
        return wrap_bexp(self)

    def __hash__(self):
        return self.raw_column.__hash__()

    def create_function(self, other, op):
        if isinstance(other, Column):
            return Function(op, [self, other], Cardinality.MANY_TO_ONE)
        else:
            return Function(op, [self, other], Cardinality.ONE_TO_ONE)

    def __add__(self, other):
        data_type = self.raw_column.node.node_type
        if data_type != BaseType.FLOAT:
            raise ColumnTypeException("float", str(data_type))
        lexp, rexp = get_arguments_for_binary_aexp(self, other)
        return AddAexp(lexp, rexp)

    def __radd__(self, other):
        return self.__add__(other)

    def __sub__(self, other):
        data_type = self.raw_column.node.node_type
        if data_type != BaseType.FLOAT:
            raise ColumnTypeException("float", str(data_type))
        lexp, rexp = get_arguments_for_binary_aexp(self, other)
        return SubAexp(lexp, rexp)

    def __rsub__(self, other):
        data_type = self.raw_column.node.node_type
        if data_type != BaseType.FLOAT:
            raise ColumnTypeException("float", str(data_type))
        lexp, rexp = get_arguments_for_binary_aexp(other, self)
        return SubAexp(lexp, rexp)

    def __mul__(self, other):
        data_type = self.raw_column.node.node_type
        if data_type != BaseType.FLOAT:
            raise ColumnTypeException("float", str(data_type))
        lexp, rexp = get_arguments_for_binary_aexp(self, other)
        return MulAexp(lexp, rexp)

    def __rmul__(self, other):
        return self.__mul__(other)

    def __truediv__(self, other):
        data_type = self.raw_column.node.node_type
        if data_type != BaseType.FLOAT:
            raise ColumnTypeException("float", str(data_type))
        lexp, rexp = get_arguments_for_binary_aexp(self, other)
        return DivAexp(lexp, rexp)

    def __rtruediv__(self, other):
        data_type = self.raw_column.node.node_type
        if data_type != BaseType.FLOAT:
            raise ColumnTypeException("float", str(data_type))
        lexp, rexp = get_arguments_for_binary_aexp(other, self)
        return DivAexp(lexp, rexp)

    def sum(self):
        data_type = self.raw_column.node.node_type
        if data_type != BaseType.FLOAT:
            raise ColumnTypeException("float", str(data_type))
        keys = self.raw_column.get_strong_keys()
        return SumAexp(keys, self)

    def max(self):
        data_type = self.raw_column.node.node_type
        if data_type != BaseType.FLOAT:
            raise ColumnTypeException("float", str(data_type))
        keys = self.raw_column.get_strong_keys()
        return MaxAexp(keys, self)

    def min(self):
        data_type = self.raw_column.node.node_type
        if data_type != BaseType.FLOAT:
            raise ColumnTypeException("float", str(data_type))
        keys = self.raw_column.get_strong_keys()
        return MinAexp(keys, self)

    def count(self):
        keys = self.raw_column.get_strong_keys()
        return CountAexp(keys, self)

    def pop(self):
        keys = self.raw_column.get_strong_keys()
        return PopExp(keys, self, self.raw_column.node.node_type)

    def get_explicit_keys(self):
        return self.raw_column.get_strong_keys()

    def get_hidden_keys(self):
        return self.raw_column.get_hidden_keys()

    def aggregate(self, function):
        return AggregationFunction(function, self)

    def apply(self, function, cardinality=Cardinality.MANY_TO_ONE):
        if cardinality == Cardinality.MANY_TO_ONE:
            return create_function(lambda c: c.apply(function))([self])
        else:
            return create_bijection(lambda c: c.apply(function))([self])
