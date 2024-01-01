from schema import BaseType


def wrap_bexp(exp):
    from tables.exp import Exp
    from tables.column import Column
    from tables.bexp import ConstBexp, Bexp, ColumnBexp
    from tables.exceptions import ColumnTypeException
    if isinstance(exp, bool):
        return ConstBexp(exp)
    elif isinstance(exp, Column):
        if exp.raw_column.node.node_type is not BaseType.BOOL:
            raise ColumnTypeException("bool", str(exp.raw_column.node.node_type))
        return ColumnBexp(exp)
    elif isinstance(exp, Exp) and exp.exp_type == BaseType.BOOL:
        return exp