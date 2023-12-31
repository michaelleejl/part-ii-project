from schema import BaseType


def wrap_aexp(exp):
    from tables.aexp import ConstAexp, Aexp, ColumnAexp
    from tables.exceptions import ColumnTypeException
    from tables.column import Column
    if isinstance(exp, float) or isinstance(exp, int):
        return ConstAexp(exp)
    elif isinstance(exp, Column):
        if exp.raw_column.node.node_type is not BaseType.FLOAT:
            raise ColumnTypeException("float", str(exp.raw_column.node.node_type))
        return ColumnAexp(exp)
    elif isinstance(exp, Aexp):
        return exp