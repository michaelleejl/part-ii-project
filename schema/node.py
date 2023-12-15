import numpy as np

from schema.helpers.is_permutation import is_permutation
from schema.helpers.is_sublist import is_sublist


class SchemaNodeNameShouldNotContainSemicolonException(Exception):
    def __init__(self, name):
        super().__init__(f"Schema node name should not contain a semicolon. Name: {name}")



class SchemaNode:
    def __init__(self, name: str, constituents: list[any] = None, cluster: str = None, graph=None):
        if ";" in name and constituents is None:
            raise SchemaNodeNameShouldNotContainSemicolonException(name)
        self.name = name
        self.constituents = constituents
        self.cluster = cluster
        self.key = (self.name, self.cluster) if self.cluster is not None else self.name
        self.graph = graph

    def prepend_id(self, val: str) -> str:
        return f"{hash(self.key)}_{val}"

    def get_key(self):
        return self.key

    @classmethod
    def product(cls, nodes: list[any]):
        atomics = []
        for node in nodes:
            cs = SchemaNode.get_constituents(node)
            atomics += cs
        name = ";".join([a.name for a in atomics])
        clusters = frozenset([n.cluster for n in nodes])
        cluster = None
        if len(clusters) == 1:
            cluster, = clusters
        constituents = atomics
        if len(constituents) == 1:
            constituents = None
        return SchemaNode(name, constituents, cluster)

    @classmethod
    def get_constituents(cls, node):
        c = node.constituents
        if c is None:
            return [node]
        else:
            return c

    @classmethod
    def is_atomic(cls, node):
        return len(SchemaNode.get_constituents(node)) == 1

    @classmethod
    def is_equivalent(cls, node1, node2, equivalence_class):
        c1 = SchemaNode.get_constituents(node1)
        c2 = SchemaNode.get_constituents(node2)
        eq = np.all(np.array(list(map(lambda x: equivalence_class.find_leader(x[0]) == equivalence_class.find_leader(x[1]), zip(c1, c2)))))
        return len(c1) == len(c2) and eq

    def __hash__(self):
        if self.constituents is None:
            return hash(self.get_key())
        else:
            return hash(frozenset(self.constituents))

    def equivalent_up_to_permutation(self, other):
        if isinstance(other, SchemaNode):
            return is_permutation(SchemaNode.get_constituents(self), SchemaNode.get_constituents(other))
        return NotImplemented

    def __eq__(self, other):
        if isinstance(other, SchemaNode):
            # check if it's atomic
            if SchemaNode.is_atomic(self):
                return SchemaNode.is_atomic(other) and self.get_key() == other.get_key()
            else:
                return SchemaNode.get_constituents(self) == SchemaNode.get_constituents(other)
        return NotImplemented

    def __le__(self, other):
        if isinstance(other, SchemaNode):
            return is_sublist(SchemaNode.get_constituents(self), SchemaNode.get_constituents(other))
        return NotImplemented

    def __lt__(self, other):
        if isinstance(other, SchemaNode):
            return (is_sublist(SchemaNode.get_constituents(self), SchemaNode.get_constituents(other))
                    and SchemaNode.get_constituents(self) != SchemaNode.get_constituents(other))
        return NotImplemented

    def __ge__(self, other):
        if isinstance(other, SchemaNode):
            return is_sublist(SchemaNode.get_constituents(other), SchemaNode.get_constituents(self))
        return NotImplemented

    def __gt__(self, other):
        if isinstance(other, SchemaNode):
            return (is_sublist(SchemaNode.get_constituents(other), SchemaNode.get_constituents(self)) and
                    SchemaNode.get_constituents(self) != SchemaNode.get_constituents(other))
        return NotImplemented

    def __repr__(self):
        if self.cluster is not None:
            return f"{self.cluster}.{self.name}"
        else:
            return self.name

    def __str__(self):
        return self.__repr__()
