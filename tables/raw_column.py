from collections import deque

from schema import SchemaNode, SchemaEdge

from enum import Enum


class ColumnType(Enum):
    KEY = 0
    VALUE = 1


class RawColumn:
    def __init__(self, name: str, node: SchemaNode, keyed_by: list[any], type: ColumnType, derivation: list[SchemaEdge] = None, table = None):
        self.name = name
        self.node = node
        self.keyed_by = keyed_by
        self.type = type
        if derivation is None:
            self.derivation = []
        self.derivation = derivation
        self.table = table

    def get_derivation(self):
        return self.derivation

    def set_derivation(self, derivation):
        self.derivation = derivation

    def __str__(self):
        return self.__repr__()

    def __repr__(self):
        return self.name

    def __hash__(self):
        return hash((self.name, tuple(self.keyed_by)))

    def __len__(self):
        return 1

    def find_leaf_keys(self):
        leaf_keys = set()
        visited = set(self.keyed_by)
        to_explore = deque()
        to_explore.extend(self.keyed_by)
        while len(to_explore) > 0:
            u = to_explore.popleft()
            if len(u.keyed_by) == 0:
                leaf_keys.add(u)
            for v in u.keyed_by:
                if v not in visited:
                    to_explore.append(v)
        return leaf_keys

    def get_explicit_keys(self):
        leaf_keys = self.find_leaf_keys()
        keys = set(self.table.keys.values())
        def is_visible_and_atomic(column):
            col_leaf_keys = column.find_leaf_keys()
            return len(set(col_leaf_keys).intersection(self.table.hidden_keys.values())) == 0

        values = set([v for v in self.table.values.values() if is_visible_and_atomic(v)])
        return leaf_keys.intersection(keys.union(values))

    def get_hidden_keys(self):
        leaf_keys = self.find_leaf_keys()
        return leaf_keys.intersection(set(self.table.hidden_keys.values()))

    @classmethod
    def assign_new_table(cls, column, table):
        return RawColumn(column.name, column.node, column.keyed_by, column.type, column.derivation, table=table)

    def __eq__(self, other):
        if isinstance(other, RawColumn):
            return self.name == other.name and self.keyed_by == other.keyed_by
        else:
            raise NotImplemented()
