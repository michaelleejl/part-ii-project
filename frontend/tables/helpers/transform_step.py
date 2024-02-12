from typing import Callable

from frontend.domain import Domain
from representation.representation import RepresentationStep


def transform_step(
    namespace, table, start, end, aggregated_over
) -> Callable[[RepresentationStep], tuple[RepresentationStep, list[Domain]]]:
    internal_namespace = namespace

    from frontend.tables import new_domain_from_schema_node

    def internal(step) -> tuple[RepresentationStep, list[Domain]]:
        from representation.representation import Traverse, Expand, EndTraversal

        if isinstance(step, Traverse) or isinstance(step, Expand):
            step_hidden_keys = step.hidden_keys
            columns = []
            for hk in step_hidden_keys:
                col = new_domain_from_schema_node(internal_namespace, hk.node)
                internal_namespace.add(col.name)
                columns += [col]
            if isinstance(step, Traverse):
                return Traverse(step.edge, columns), columns
            else:
                return (
                    Expand(
                        step.start_node,
                        step.end_node,
                        step.indices,
                        step.hidden_keys,
                    ),
                    columns,
                )
        elif isinstance(step, EndTraversal):
            return (
                EndTraversal(end),
                [],
            )
        else:
            return step, []

    return internal
