from representation.representation import (
    RepresentationStep,
    StartTraversal,
    EndTraversal,
)
from typing import Callable


def invert_representation(
    representation: list[RepresentationStep],
    namespace: set[str],
    naming_function: Callable[[set[str], str], str],
) -> list[RepresentationStep]:
    """
    Inverts a representation. If the original representation describes how to derive Y from X, the inverted
    representation describes how to derive X from Y.

    Args:
        representation (list[RepresentationStep]): The representation to invert
        namespace (set[str]): The namespace of the representation
        naming_function (Callable[[set[str], str], str]): The naming function for naming hidden keys

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
            stack = [command.invert()] + stack
    result += stack
    return result
