from frontend.domain import Domain
from frontend.tables.helpers.carry_keys_through_representation import (
    carry_keys_through_representation,
)
from representation.representation import (
    RepresentationStep,
    Traverse,
    Expand,
    StartTraversal,
    EndTraversal,
)
from schema.helpers.find_index import find_index
from schema.node import SchemaNode


def find_index_of_instruction_causing_hidden_key(
    key: Domain, representation: list[RepresentationStep]
) -> int | None:
    for i, step in enumerate(representation):
        if step.name == "TRV":
            assert isinstance(step, Traverse)
            if key in step.get_hidden_keys():
                return i
        elif step.name == "EXP":
            assert isinstance(step, Expand)
            if key in step.get_hidden_keys():
                return i
    return None


def show_key_in_representation_segment(
    key: Domain, representation: list[RepresentationStep]
) -> tuple[list[RepresentationStep], int]:
    index = find_index_of_instruction_causing_hidden_key(key, representation)
    if index is None:
        return representation, -1
    offending = representation[index]
    stt = [
        i for i, x in enumerate(representation[:index]) if isinstance(x, StartTraversal)
    ]
    ent = [
        index + 1 + i
        for i, x in enumerate(representation[index + 1 :])
        if isinstance(x, EndTraversal)
    ]

    last_stt = stt[-1]
    first_ent = ent[0]

    prefix = representation[:last_stt] + carry_keys_through_representation(
        [key], representation[last_stt:index]
    )
    suffix = representation[index + 1 :]

    if isinstance(offending, Traverse):
        return prefix + [Traverse(offending.edge.uncurry(key))] + suffix, first_ent

    if isinstance(offending, Expand):
        if len(offending.hidden_keys) == 1:
            return prefix + suffix, first_ent
        else:
            start = SchemaNode.product([offending.start_node, key.node])
            hidden_key_idx = find_index(key, offending.hidden_keys)
            masked_indices = [
                i
                for i in range(len(SchemaNode.get_constituents(offending.end_node)))
                if i not in set(offending.indices)
            ]
            idx_to_add = masked_indices[hidden_key_idx]
            new_hidden_keys = (
                offending.hidden_keys[:hidden_key_idx]
                + offending.hidden_keys[hidden_key_idx + 1 :]
            )
            exp = Expand(
                start,
                offending.end_node,
                offending.indices + [idx_to_add],
                new_hidden_keys,
            )
            return prefix + [exp] + suffix, first_ent


def show_key_in_representation(
    key: Domain, representation: list[RepresentationStep]
) -> list[RepresentationStep]:
    steps = representation
    index = 0
    while 0 <= index <= len(representation):
        steps[index:], index = show_key_in_representation_segment(key, steps[index:])
    return steps
