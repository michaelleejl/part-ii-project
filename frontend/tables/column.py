from schema.base_types import BaseType
from exp.aexp import (
    ColumnAexp,
    AddAexp,
    SubAexp,
    MulAexp,
    DivAexp,
    SumAexp,
    MaxAexp,
    MinAexp,
)
from exp.bexp import (
    EqualityBexp,
    NotBexp,
    LessThanBexp,
    NABexp,
    OrBexp,
    AndBexp,
    ColumnBexp,
    AnyBexp,
    AllBexp,
)
from frontend.tables.exceptions import ColumnTypeException
from exp.exp import PopExp, CountExp, ExtendExp, MaskExp
from exp.helpers.wrap_aexp import wrap_aexp
from exp.helpers.wrap_bexp import wrap_bexp
from exp.helpers.wrap_sexp import wrap_sexp


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
    def __init__(self, node):
        self.name = node.get_name()
        self.node = node

    # Boolean expressions

    def get_domain(self):
        assert len(self.node.domains) == 1
        return self.node.domains[0]

    def get_schema_node(self):
        return self.get_domain().node

    def get_type(self):
        return self.get_schema_node().node_type

    def __eq__(self, other) -> EqualityBexp:
        data_type = self.get_type()
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
        return NotBexp(self == other)

    def __lt__(self, other) -> LessThanBexp:
        data_type = self.get_type()
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
        data_type = self.get_type()
        if data_type != BaseType.BOOL:
            raise ColumnTypeException("bool", str(data_type))
        lexp, rexp = get_arguments_for_binary_bexp(self, other)
        return AndBexp(lexp, rexp)

    def __rand__(self, other):
        self.__and__(other)

    def __or__(self, other):
        data_type = self.get_type()
        if data_type != BaseType.BOOL:
            raise ColumnTypeException("bool", str(data_type))
        lexp, rexp = get_arguments_for_binary_bexp(self, other)
        return OrBexp(lexp, rexp)

    def __ror__(self, other):
        self.__or__(other)

    def __invert__(self):
        data_type = self.get_type()
        if data_type != BaseType.BOOL:
            raise ColumnTypeException("bool", str(data_type))
        exp = wrap_bexp(self)
        return NotBexp(exp)

    def any(self):
        data_type = self.get_type()
        if data_type != BaseType.BOOL:
            raise ColumnTypeException("bool", str(data_type))
        keys = self.get_strong_keys()
        hids = self.get_hidden_keys()
        return AnyBexp(keys, hids, self.get_domain())

    def all(self):
        data_type = self.get_type()
        if data_type != BaseType.BOOL:
            raise ColumnTypeException("bool", str(data_type))
        keys = self.get_strong_keys()
        hids = self.get_hidden_keys()
        return AllBexp(keys, hids, self.get_domain())

    def isnull(self) -> NABexp:
        data_type = self.get_type()
        if data_type == BaseType.FLOAT:
            exp = ColumnAexp(self.get_domain())
        elif data_type == BaseType.BOOL:
            exp = ColumnBexp(self.get_domain())
        else:
            raise NotImplemented()
        return NABexp(exp)

    def isnotnull(self) -> NotBexp:
        return NotBexp(self.isnull())

    def to_bexp(self):
        return wrap_bexp(self)

    def __hash__(self):
        return self.node.__hash__()

    def __add__(self, other):
        data_type = self.get_type()
        if data_type != BaseType.FLOAT:
            raise ColumnTypeException("float", str(data_type))
        lexp, rexp = get_arguments_for_binary_aexp(self, other)
        return AddAexp(lexp, rexp)

    def __radd__(self, other):
        return self.__add__(other)

    def __sub__(self, other):
        data_type = self.get_type()
        if data_type != BaseType.FLOAT:
            raise ColumnTypeException("float", str(data_type))
        lexp, rexp = get_arguments_for_binary_aexp(self, other)
        return SubAexp(lexp, rexp)

    def __rsub__(self, other):
        data_type = self.get_type()
        if data_type != BaseType.FLOAT:
            raise ColumnTypeException("float", str(data_type))
        lexp, rexp = get_arguments_for_binary_aexp(other, self)
        return SubAexp(lexp, rexp)

    def __mul__(self, other):
        data_type = self.get_type()
        if data_type != BaseType.FLOAT:
            raise ColumnTypeException("float", str(data_type))
        lexp, rexp = get_arguments_for_binary_aexp(self, other)
        return MulAexp(lexp, rexp)

    def __rmul__(self, other):
        return self.__mul__(other)

    def __truediv__(self, other):
        data_type = self.get_type()
        if data_type != BaseType.FLOAT:
            raise ColumnTypeException("float", str(data_type))
        lexp, rexp = get_arguments_for_binary_aexp(self, other)
        return DivAexp(lexp, rexp)

    def __rtruediv__(self, other):
        data_type = self.get_type()
        if data_type != BaseType.FLOAT:
            raise ColumnTypeException("float", str(data_type))
        lexp, rexp = get_arguments_for_binary_aexp(other, self)
        return DivAexp(lexp, rexp)

    def sum(self):
        data_type = self.get_type()
        if data_type != BaseType.FLOAT:
            raise ColumnTypeException("float", str(data_type))
        keys = self.node.get_strong_keys()
        hids = self.get_hidden_keys()
        return SumAexp(keys, hids, self.get_domain())

    def max(self):
        data_type = self.get_type()
        if data_type != BaseType.FLOAT:
            raise ColumnTypeException("float", str(data_type))
        keys = self.get_strong_keys()
        hids = self.get_hidden_keys()
        return MaxAexp(keys, hids, self.get_domain())

    def min(self):
        data_type = self.get_type()
        if data_type != BaseType.FLOAT:
            raise ColumnTypeException("float", str(data_type))
        keys = self.get_strong_keys()
        hids = self.get_hidden_keys()
        return MinAexp(keys, hids, self.get_domain())

    def count(self):
        keys = self.get_strong_keys()
        hids = self.get_hidden_keys()
        return CountExp(keys, hids, self.get_domain(), self.node.exp_type)

    def pop(self):
        keys = self.node.get_strong_keys()
        hids = self.get_hidden_keys()
        return PopExp(keys, hids, self.get_domain(), self.get_type())

    def get_strong_keys(self):
        return self.node.get_strong_keys()

    def get_hidden_keys(self):
        return self.node.get_hidden_keys()

    def wrap_function(self, expr):
        node_type = self.get_type()

        match node_type:
            case BaseType.FLOAT:
                expr = wrap_aexp(expr)
            case BaseType.BOOL:
                expr = wrap_bexp(expr)
            case BaseType.STRING:
                expr = wrap_sexp(expr)

        return expr

    def extend(self, with_function):
        assert self.node.is_val_column()
        expr = self.wrap_function(with_function)
        assert self.get_type() == expr.exp_type
        return ExtendExp([], self.get_domain(), expr, expr.exp_type)

    def mask(self, with_condition):
        expr = self.wrap_function(with_condition)
        assert self.get_type() == expr.exp_type
        return MaskExp([], self.get_domain(), expr, expr.exp_type)
