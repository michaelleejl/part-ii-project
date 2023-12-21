from collections import deque

from schema import SchemaNode, SchemaEdge, Cardinality

from enum import Enum


class ColumnType(Enum):
    KEY = 0
    VALUE = 1


class RawColumn:
    def __init__(self, name: str, node: SchemaNode, strong_keys: list[any], hidden_keys: list[any],
                 is_strong_key_for_self: bool, cardinality: Cardinality | None, type: ColumnType,
                 table=None):
        self.name = name
        self.node = node
        self.strong_keys = strong_keys
        self.hidden_keys = hidden_keys
        self.is_strong_key_for_self = is_strong_key_for_self
        self.cardinality = cardinality
        self.type = type
        self.table = table

    def __str__(self):
        return self.__repr__()

    def __repr__(self):
        return self.name

    def __hash__(self):
        return hash((self.name, tuple(self.strong_keys), tuple(self.hidden_keys)))

    def __len__(self):
        return 1

    def set_cardinality(self, cardinality):
        self.cardinality = cardinality

    def set_strong_keys(self, strong_keys):
        self.strong_keys = strong_keys

    def set_hidden_keys(self, hidden_keys):
        self.hidden_keys = hidden_keys

    def find_leaf_keys(self):
        return self.strong_keys + self.hidden_keys

    def get_strong_keys(self):
        if self.is_strong_key_for_self:
            return self.strong_keys + [self]
        else:
            return self.strong_keys

    def get_hidden_keys(self):
        return self.hidden_keys

    @classmethod
    def assign_new_table(cls, column, table):
        return RawColumn(column.name, column.node, column.strong_keys, column.hidden_keys,
                         column.is_strong_key_for_self, column.cardinality, column.type, table=table)

    def __eq__(self, other):
        if isinstance(other, RawColumn):
            return (self.name == other.name
                    and self.strong_keys == other.strong_keys
                    and self.hidden_keys == other.hidden_keys)
        else:
            raise NotImplemented()
