import copy
from typing import TypeVar, Generic

from schema.helpers.find_index import find_index

T = TypeVar("T")

class OrderedSet(Generic[T]):
    def __init__(self, items=None):
        self.item_list = []
        self.item_set = frozenset([])
        if items is None:
            self.item_set = frozenset()
            self.item_list = []
        else:
            for item in items:
                if item not in self.item_set:
                    self.item_set |= {item}
                    self.item_list += [item]

    def append(self, item):
        new_list = self.item_list
        if item not in self.item_set:
            return OrderedSet(new_list + [item])
        else:
            return OrderedSet(new_list)

    def append_all(self, items):
        new_set = OrderedSet(self.item_list)
        for item in items:
            new_set = new_set.append(item)
        return new_set

    def find_item(self, item):
        idx = find_index(item, self.item_list)
        if idx >= 0:
            return self.item_list[idx]

    def find_item_with_idx(self, item):
        idx = find_index(item, self.item_list)
        if idx >= 0:
            return idx, self.item_list[idx]

    def remove(self, item):
        if self.find_item_with_idx(item) is not None:
            idx, to_remove = self.find_item_with_idx(item)
            if to_remove is not None:
                return OrderedSet(self.item_list[:idx] + self.item_list[idx + 1 :])
            else:
                return OrderedSet(self.item_list)

    def remove_all(self, items):
        new_set = OrderedSet(self.item_list)
        for item in items:
            new_set = new_set.remove(item)
        return new_set

    def union(self, other):
        if isinstance(other, OrderedSet):
            return self.append_all(other.item_list)
        else:
            raise NotImplemented()

    def __len__(self):
        return len(self.item_list)

    def __iter__(self):
        return self.item_list.__iter__()

    def __getitem__(self, item):
        return self.item_list[item]

    def __repr__(self):
        return self.item_list.__repr__()

    def __copy__(self):
        return self.copy()

    def __contains__(self, item):
        return item in self.item_set

    def to_list(self):
        return copy.copy(self.item_list)

    def copy(self):
        return OrderedSet([i for i in self.item_list])
