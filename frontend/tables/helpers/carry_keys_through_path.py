from schema.node import SchemaNode
from representation.representation import *


def carry_keys_through_representation(
    representation: list[RepresentationStep], keys: list[Domain]
):
    result = []
    for command in representation:
        if isinstance(command, StartTraversal):
            start_node = SchemaNode.product(
                [n.node for n in command.start_columns + keys]
            )
            end_node = SchemaNode.product([n.node for n in command.start_columns])
            indices = list(range(len(command.start_columns)))
            result += [
                StartTraversal(command.start_columns + keys),
                Project(start_node, end_node, indices),
            ]
        elif isinstance(command, EndTraversal):
            result += [EndTraversal(command.end_columns)]
        else:
            result += [command]
    return result
