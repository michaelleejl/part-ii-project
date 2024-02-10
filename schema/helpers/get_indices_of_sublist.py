from typing import TypeVar

T = TypeVar("T")


class SubListMustBeFullyContainedInList(Exception):
    def __init__(self, sublist, list):
        super().__init__(f"Sublist {sublist} is not fully contained in list {list}")


def get_indices_of_sublist(sublist: list[T], in_list: list[T]) -> list[int]:
    """
    Get the indices of the sublist in the list

    Args:
        sublist (list[T]): The sublist
        in_list (list[T]): The list

    Returns:
        list[int]: The indices of the sublist in the list
    """
    idxs = []
    if len(sublist) > len(in_list):
        raise SubListMustBeFullyContainedInList(sublist, in_list)
    i = 0
    j = 0
    while i < len(sublist):
        if j >= len(in_list):
            raise SubListMustBeFullyContainedInList(sublist, in_list)
        if sublist[i] == in_list[j]:
            idxs += [j]
            i += 1
        j += 1
    return idxs
