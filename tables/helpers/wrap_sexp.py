from schema import BaseType


def wrap_sexp(exp):
    from tables.exp import Exp
    from tables.column import Column
    from tables.sexp import ConstSexp, ColumnSexp
    from tables.exceptions import ColumnTypeException

    if isinstance(exp, str):
        return ConstSexp(exp)
    elif isinstance(exp, Column):
        if exp.get_schema_node().node_type is not BaseType.STRING:
            raise ColumnTypeException("string", str(exp.get_schema_node().node_type))
        return ColumnSexp(exp.node.domains[0])
    elif isinstance(exp, Exp) and exp.exp_type == BaseType.STRING:
        return exp
