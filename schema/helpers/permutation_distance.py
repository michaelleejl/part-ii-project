import numpy as np

from schema.helpers.is_permutation import is_permutation


def permutation_distance(list1: list[any], list2: list[any]):
    assert is_permutation(list1, list2)
    return np.sum(np.array(list1) == np.array(list2)) / 2