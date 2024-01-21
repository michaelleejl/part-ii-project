
from schema import Cardinality, AtomicNode

from enum import Enum

class Domain:
    def __init__(self, name: str, node: AtomicNode):
        self.name = name
        self.node = node

    def __str__(self):
        return self.__repr__()

    def __repr__(self):
        return self.name

    def __hash__(self):
        return hash((self.name, self.node))

    def __len__(self):
        return 1

    def __eq__(self, other):
        if isinstance(other, Domain):
            return self.__hash__() == other.__hash__()
        else:
            raise NotImplemented()
