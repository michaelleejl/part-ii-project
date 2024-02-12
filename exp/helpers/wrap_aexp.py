def wrap_aexp(exp):
    from exp.exp import Exp
    from exp.aexp import ConstAexp, ColumnAexp
    from frontend.tables.exceptions import ColumnTypeException
    from frontend.tables.column import Column
    from schema.base_types import BaseType

    if isinstance(exp, float) or isinstance(exp, int):
        return ConstAexp(exp)
    elif isinstance(exp, Column):
        if exp.get_schema_node().node_type is not BaseType.FLOAT:
            raise ColumnTypeException("float", str(exp.get_schema_node().node_type))
        return ColumnAexp(exp.node.domains[0])
    elif isinstance(exp, Exp) and exp.exp_type == BaseType.FLOAT:
        return exp
