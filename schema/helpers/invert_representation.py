from schema import SchemaEdge, SchemaNode
from tables.internal_representation import *


def invert_command(command: RepresentationStep):
    match command.name:
        case "PSH":
            return Push()
        case "POP":
            return Pop()
        case "TRV":
            assert isinstance(command, Traverse)
            edge = command.edge
            rev = SchemaEdge.invert(edge)
            return Traverse(rev)
        case "EQU":
            return command
        case "PRJ":
            assert isinstance(command, Project)
            nodes = SchemaNode.get_constituents(command.start_node)
            hidden_keys = [n for (i, n) in enumerate(nodes) if i not in set(command.indices)]
            return Expand(command.end_node, command.start_node, command.indices, hidden_keys)
        case "EXP":
            assert isinstance(command, Expand)
            return Project(command.end_node, command.start_node, command.indices)
        case "RNM":
            return command
        case "FLT":
            return command
        case "SRT":
            return command
        case _:
            return command


def invert_representation(representation: list[RepresentationStep]):
    STT: StartTraversal | None = None
    stack = []
    result = []
    for command in representation:
        if isinstance(command, StartTraversal):
            result += stack
            stack = []
            STT = command
        elif isinstance(command, EndTraversal):
            assert STT is not None
            start = StartTraversal(command.end_columns)
            end = EndTraversal(command.end_columns, command.start_columns)
            result += [start] + stack + [end]
            stack = []
        else:
            stack += [invert_command(command)]
    result += stack
    return result
