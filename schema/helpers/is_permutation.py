from collections import Counter


def is_permutation(list1: list[any], list2: list[any]):
    return Counter(list1) == Counter(list2)