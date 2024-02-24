from representation.representation import (
    RepresentationStep,
    StartTraversal,
    EndTraversal,
)
from typing import Callable


def invert_representation(
    representation: list[RepresentationStep],
    namespace: frozenset[str],
) -> tuple[list[RepresentationStep], frozenset[str]]:
    """
    Inverts a representation. If the original representation describes how to derive Y from X, the inverted
    representation describes how to derive X from Y.

    Args:
        representation (list[RepresentationStep]): The representation to invert
        namespace (set[str]): The namespace of the representation

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
            end = EndTraversal(STT.start_columns)
            result += [start] + stack + [end]
            stack = []
        else:
            new_command, namespace = command.invert(namespace)
            stack = [new_command] + stack
    result += stack
    return result, namespace
