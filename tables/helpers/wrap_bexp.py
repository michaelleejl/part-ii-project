from schema import BaseType


def wrap_bexp(exp):
    from tables.exp import Exp
    from tables.column import Column
    from tables.bexp import ConstBexp, Bexp, ColumnBexp
    from tables.exceptions import ColumnTypeException
    if isinstance(exp, bool):
        return ConstBexp(exp)
    elif isinstance(exp, Column):
        if exp.get_schema_node().node_type is not BaseType.BOOL:
            raise ColumnTypeException("bool", str(exp.get_schema_node().node_type))
        return ColumnBexp(exp.node.domains[0])
    elif isinstance(exp, Exp) and exp.exp_type == BaseType.BOOL:
        return exp