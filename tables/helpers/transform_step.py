from typing import Callable, Any, Tuple, List

from tables.domain import Domain
from tables.internal_representation import RepresentationStep


def transform_step(namespace, table, start, end, aggregated_over) -> Callable[[RepresentationStep], tuple[RepresentationStep, list[Domain]]]:
    internal_namespace = namespace

    def internal(step) -> tuple[RepresentationStep, list[Domain]]:
        from tables.internal_representation import Traverse, Expand, EndTraversal
        if isinstance(step, Traverse) or isinstance(step, Expand):
            step_hidden_keys = step.hidden_keys
            columns = []
            for hk in step_hidden_keys:
                col = table.new_col_from_node(internal_namespace, hk)
                internal_namespace.add(col.name)
                columns += [col]
            if isinstance(step, Traverse):
                return Traverse(step.edge, columns), columns
            else:
                return Expand(step.start_node, step.end_node, step.indices, step.hidden_keys, columns), columns
        elif isinstance(step, EndTraversal):
            return EndTraversal([c for c in start if c not in set(aggregated_over)], end), []
        else:
            return step, []

    return internal