import pandas as pd


class SchemaNode:
    def __init__(self, name, constituents: frozenset[any] = None):
        self.name = name
        self.key = self.name
        if constituents is None:
            self.constituents = frozenset([self])
        else:
            self.constituents = constituents

    def prepend_id(self, val: str) -> str:
        return f"{hash(self.key)}_{val}"

    def get_key(self):
        return self.key

    @classmethod
    def product(cls, nodes: frozenset[any]):
        atomics = frozenset()
        for node in nodes:
            atomics = atomics.union(node.constituents)
        name = "; ".join([a.name for a in atomics])
        constituents = atomics
        return SchemaNode(name, constituents)

    def __hash__(self):
        if len(self.constituents) == 1:
            return hash(self.get_key())
        else:
            return hash(self.constituents)

    def __eq__(self, other):
        if isinstance(other, SchemaNode):
            # check if it's atomic
            if len(self.constituents) == 1:
                return len(other.constituents) == 1 and self.get_key() == other.get_key()
            else:
                return self.constituents == other.constituents
        return NotImplemented

    def __le__(self, other):
        if isinstance(other, SchemaNode):
            return self.constituents <= other.constituents
        return NotImplemented

    def __lt__(self, other):
        if isinstance(other, SchemaNode):
            return self.constituents < other.constituents
        return NotImplemented

    def __ge__(self, other):
        if isinstance(other, SchemaNode):
            return self.constituents >= other.constituents
        return NotImplemented

    def __gt__(self, other):
        if isinstance(other, SchemaNode):
            return self.constituents > other.constituents
        return NotImplemented

    def __repr__(self):
        return f"{self.name}"

    def __str__(self):
        return f"{self.name}"
