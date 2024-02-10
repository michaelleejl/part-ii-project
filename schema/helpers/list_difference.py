from typing import TypeVar

T = TypeVar("T")


def list_difference(list1: list[T], list2: list[T]) -> list[T]:
    """
    Returns all elements in list1 that are not in list2

    Args:
        list1 (list[T]): The first list
        list2 (list[T]): The second list

    Returns:
        list[T]: The elements in list1 that are not in list2
    """

    i = 0
    j = 0
    diff = []
    while i < len(list1) and j < len(list2):
        a = list1[i]
        b = list2[j]
        if a != b:
            diff += [a]
            i += 1
        else:
            i += 1
            j += 1
    return diff + list1[i:]
