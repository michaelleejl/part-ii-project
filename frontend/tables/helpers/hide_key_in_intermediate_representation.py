from schema import SchemaNode
from schema.helpers.find_index import find_index
from representation.representation import *


def hide_key_in_intermediate_representation(
    representation: list[RepresentationStep], key: Domain
):
    steps = []
    for step in representation:
        if step.name == "STT":
            assert isinstance(step, StartTraversal)
            columns = step.start_columns
            idx = find_index(key, columns)
            if idx >= 0:
                start_columns = columns[:idx] + columns[idx + 1 :]
                start_node = SchemaNode.product([d.node for d in start_columns])
                end_columns = columns
                end_node = SchemaNode.product([d.node for d in end_columns])
                indices = [i for i in range(len(columns)) if i != idx]
                steps += [
                    StartTraversal(start_columns),
                    Expand(start_node, end_node, indices, [key]),
                    EndTraversal(end_columns),
                ]
        steps += [step]
    return steps
