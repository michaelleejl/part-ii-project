from representation.domain import Domain
from representation.representation import RepresentationStep


def rename_key_in_representation_step(
    step, namespace: frozenset[str]
) -> tuple[RepresentationStep, list[Domain], frozenset[str]]:
    from frontend.tables.table import new_domain_from_schema_node
    from representation.representation import Traverse, Expand

    internal_namespace = namespace

    if isinstance(step, Traverse) or isinstance(step, Expand):
        step_hidden_keys = step.get_hidden_keys()
        new_hks = []
        for hk in step_hidden_keys:
            new_hk = new_domain_from_schema_node(internal_namespace, hk.node, hk.name)
            internal_namespace |= {new_hk.name}
            new_hks += [new_hk]

        if isinstance(step, Traverse):
            new_edge = step.edge.replace_hidden_keys(new_hks)
            return Traverse(new_edge), new_hks, frozenset(internal_namespace)
        else:
            return (
                Expand(
                    step.start_node,
                    step.end_node,
                    step.indices,
                    new_hks,
                ),
                new_hks,
                frozenset(internal_namespace),
            )
    else:
        return step, [], frozenset(internal_namespace)


def rename_hidden_keys_in_representation(
    namespace: frozenset[str], representation: list[RepresentationStep]
) -> [tuple[RepresentationStep, list[Domain], frozenset[str]]]:
    namespace = frozenset(namespace)
    new_representation = []
    hidden_keys = []
    for step in representation:
        step, hks, namespace = rename_key_in_representation_step(step, namespace)
        new_representation += [step]
        hidden_keys += hks
    return new_representation, hidden_keys, namespace
