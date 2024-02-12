from typing import TypeVar

T = TypeVar("T")


def binary_search(l: list[T], v: T) -> T:
    low = 0
    high = len(l)
    while low < high:
        mid = (low + high) // 2
        if v < l[mid]:
            high = mid
        elif v > l[mid]:
            low = mid + 1
        else:
            return mid
    return low
