from representation.representation import *


def invert_representation(
    representation: list[RepresentationStep],
) -> list[RepresentationStep]:
    """
    Inverts a representation. If the original representation describes how to derive Y from X, the inverted
    representation describes how to derive X from Y.

    Args:
        representation (list[RepresentationStep]): The representation to invert

    Returns:
        list[RepresentationStep]: The inverted representation
    """
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
            stack = [command.invert()] + stack
    result += stack
    return result
