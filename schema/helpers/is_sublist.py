from typing import TypeVar

from schema.helpers.find_index import find_index

T = TypeVar("T")


def is_sublist(list1: list[T], list2: list[T]) -> bool:
    """
    Returns True if list1 is a sublist of list2

    Args:
        list1 (list[T]): The potential sublist
        list2 (list[T]): The list

    Returns:
        bool: True if list1 is a sublist of list2, False otherwise
    """
    if len(list1) > len(list2):
        return False
    start_index = 0
    for item in list1:
        new_index = find_index(item, list2[start_index:]) + start_index
        if start_index > new_index:
            return False
        start_index = new_index + 1
    return True
