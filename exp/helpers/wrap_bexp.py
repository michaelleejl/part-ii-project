def wrap_bexp(exp):
    from exp.exp import Exp
    from frontend.tables.column import Column
    from exp.bexp import ConstBexp, ColumnBexp
    from frontend.tables.exceptions import ColumnTypeException
    from schema.base_types import BaseType

    if isinstance(exp, bool):
        return ConstBexp(exp)
    elif isinstance(exp, Column):
        if exp.get_schema_node().node_type is not BaseType.BOOL:
            raise ColumnTypeException("bool", str(exp.get_schema_node().node_type))
        return ColumnBexp(exp.node.domains[0])
    elif isinstance(exp, Exp) and exp.exp_type == BaseType.BOOL:
        return exp
