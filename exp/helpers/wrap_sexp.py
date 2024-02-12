def wrap_sexp(exp):
    from exp.exp import Exp
    from frontend.tables.column import Column
    from exp.sexp import ConstSexp, ColumnSexp
    from frontend.tables.exceptions import ColumnTypeException
    from schema.base_types import BaseType

    if isinstance(exp, str):
        return ConstSexp(exp)
    elif isinstance(exp, Column):
        if exp.get_schema_node().node_type is not BaseType.STRING:
            raise ColumnTypeException("string", str(exp.get_schema_node().node_type))
        return ColumnSexp(exp.node.domains[0])
    elif isinstance(exp, Exp) and exp.exp_type == BaseType.STRING:
        return exp
