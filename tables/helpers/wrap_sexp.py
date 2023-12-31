from schema import BaseType


def wrap_sexp(exp):
    from tables.column import Column
    from tables.sexp import ConstSexp, Sexp, ColumnSexp
    from tables.exceptions import ColumnTypeException
    if isinstance(exp, str):
        return ConstSexp(exp)
    elif isinstance(exp, Column):
        if exp.raw_column.node.node_type is not BaseType.STRING:
            raise ColumnTypeException("string", str(exp.raw_column.node.node_type))
        return ColumnSexp(exp)
    elif isinstance(exp, Sexp):
        return exp