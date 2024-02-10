from typing import TypeVar

T = TypeVar("T")


def find_index(item: T, in_list: list[T]) -> int:
    """
    Finds the index of an item in a list, returning -1 if the item is not in the list

    Args:
        item: The item to find
        in_list: The list to search

    Returns:
        int: The index of the item in the list, or -1 if the item is not in the list
    """
    try:
        return in_list.index(item)
    except ValueError:
        return -1
