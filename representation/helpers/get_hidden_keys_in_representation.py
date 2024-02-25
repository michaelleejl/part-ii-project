from frontend.domain import Domain
from representation.representation import RepresentationStep


def get_hidden_keys_in_representation(
    representation: list[RepresentationStep],
) -> list[Domain]:
    """
    Returns the hidden keys in a representation

    Args:
        representation (list[RepresentationStep]): The representation to analyze

    Returns:
        list[str]: The hidden keys in the representation
    """
    hidden_keys_set = set()
    hidden_keys = []
    for step in representation:
        filtered = [hk for hk in step.get_hidden_keys() if hk not in hidden_keys_set]
        hidden_keys += filtered
        hidden_keys_set |= set(step.get_hidden_keys())
    return hidden_keys
