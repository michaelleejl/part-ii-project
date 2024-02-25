from schema.node import SchemaNode
from representation.representation import *


def carry_keys_through_representation(
    keys: list[Domain], representation: list[RepresentationStep]
):
    nodes = [key.node for key in keys]
    result = []
    for command in representation:
        if isinstance(command, StartTraversal):
            result += [
                StartTraversal(command.start_columns + keys),
            ]

        elif isinstance(command, EndTraversal):
            assert False

        elif isinstance(command, Traverse):
            result += [Traverse(command.edge.carry_multiple(keys))]

        elif isinstance(command, Project):
            from_node = SchemaNode.product([command.start_node, *nodes])
            to_node = SchemaNode.product([command.end_node, *nodes])

            n = len(SchemaNode.get_constituents(command.start_node))
            m = len(nodes)

            new_indices = [i for i in range(n, n + m)]

            result += [Project(from_node, to_node, new_indices)]

        elif isinstance(command, Expand):
            from_node = SchemaNode.product([command.start_node, *nodes])
            to_node = SchemaNode.product([command.end_node, *nodes])

            n = len(SchemaNode.get_constituents(command.end_node))
            m = len(nodes)

            new_indices = [i for i in range(n, n + m)]
            result += [
                Expand(
                    from_node,
                    to_node,
                    command.indices + new_indices,
                    command.hidden_keys,
                )
            ]

        elif isinstance(command, Equate):
            from_node = SchemaNode.product([command.start_node, *nodes])
            to_node = SchemaNode.product([command.end_node, *nodes])
            result += [Equate(from_node, to_node)]

        else:
            result += [command]

    return result
