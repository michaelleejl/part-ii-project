import abc
import uuid

import numpy as np

from schema.base_types import BaseType


class SchemaNodeNameShouldNotContainSemicolonException(Exception):
    def __init__(self, name):
        super().__init__(f"Schema node name should not contain a semicolon. Name: {name}")


class SchemaNode(abc.ABC):

    @classmethod
    def product(cls, nodes: list[any]):
        atomics = []
        for node in nodes:
            cs = SchemaNode.get_constituents(node)
            atomics += cs
        constituents = atomics
        if len(constituents) == 1:
            node, = atomics
            return node
        else:
            return ProductNode(constituents)

    @classmethod
    def get_constituents(cls, node):
        if isinstance(node, AtomicNode) or isinstance(node, SchemaClass):
            return [node]
        else:
            return node.constituents

    @classmethod
    def is_atomic(cls, node):
        return isinstance(node, AtomicNode)

    @classmethod
    def is_equivalent(cls, node1, node2, equivalence_class):
        c1 = SchemaNode.get_constituents(node1)
        c2 = SchemaNode.get_constituents(node2)
        eq = np.all(np.array(list(
            map(lambda x: equivalence_class.find_leader(x[0]) == equivalence_class.find_leader(x[1]), zip(c1, c2)))))
        return len(c1) == len(c2) and eq

    @abc.abstractmethod
    def __eq__(self, other):
        pass

    @abc.abstractmethod
    def __le__(self, other):
        pass

    @abc.abstractmethod
    def __lt__(self, other):
        pass

    @abc.abstractmethod
    def __ge__(self, other):
        pass

    @abc.abstractmethod
    def __gt__(self, other):
        pass


class UnknownBaseClassException(Exception):
    def __init__(self, base_class):
        super().__init__(f"Base class must be one of Object, Float, Bool, or String, not {base_class}")


class AtomicNode(SchemaNode):
    def __init__(self, name: str, node_type: BaseType):
        super().__init__()
        self.node_type = node_type
        self.name = name
        self.id = uuid.uuid4()
        self.id_prefix = 3

    @classmethod
    def clone(cls, node, name: str | None):
        if name is None:
            name = node.name
        return AtomicNode(name, node.node_type)

    def get_id(self):
        return self.id

    def __hash__(self):
        return hash(self.id)

    def __eq__(self, other):
        if isinstance(other, SchemaNode):
            if isinstance(other, AtomicNode):
                return self.id == other.id
            elif isinstance(other, ProductNode):
                return False
            return False
        else:
            raise NotImplemented()

    def __le__(self, other):
        if isinstance(other, SchemaNode):
            if isinstance(other, AtomicNode):
                return self == other
            elif isinstance(other, ProductNode):
                return self in other.constituents
            return False
        raise NotImplemented()

    def __lt__(self, other):
        if isinstance(other, SchemaNode):
            if isinstance(other, AtomicNode):
                return False
            elif isinstance(other, ProductNode):
                return self in other.constituents
            return False
        raise NotImplemented()

    def __ge__(self, other):
        if isinstance(other, SchemaNode):
            if isinstance(other, AtomicNode):
                return self == other
            elif isinstance(other, ProductNode):
                return False
            return False
        raise NotImplemented()

    def __gt__(self, other):
        if isinstance(other, SchemaNode):
            return False
        raise NotImplemented()

    def __repr__(self):
        return f"{self.name} <{str(self.id)[:self.id_prefix]}>"

    def __str__(self):
        return self.name


class ProductNodeShouldHaveAtLeastTwoConstituentsException(Exception):
    def __init__(self, constituents):
        super().__init__(f"Product node should have at least two constituents. Constituents: {constituents}")


class ProductNode(SchemaNode):

    def __init__(self, constituents: list[AtomicNode]):
        super().__init__()
        if len(constituents) <= 1:
            raise ProductNodeShouldHaveAtLeastTwoConstituentsException(constituents)
        self.constituents = constituents

    def __hash__(self):
        return hash(tuple(self.constituents))

    def __eq__(self, other):
        if isinstance(other, SchemaNode):
            if isinstance(other, AtomicNode):
                return False
            elif isinstance(other, ProductNode):
                return self.constituents == other.constituents
            return False
        raise NotImplemented()

    def __lt__(self, other):
        if isinstance(other, SchemaNode):
            if isinstance(other, AtomicNode):
                return False
            elif isinstance(other, ProductNode):
                from schema.helpers.is_sublist import is_sublist
                return is_sublist(self.constituents, other.constituents) and self != other
            return False
        raise NotImplemented()

    def __le__(self, other):
        if isinstance(other, SchemaNode):
            if isinstance(other, AtomicNode):
                return False
            elif isinstance(other, ProductNode):
                from schema.helpers.is_sublist import is_sublist
                return is_sublist(self.constituents, other.constituents)
            return False
        raise NotImplemented()

    def __gt__(self, other):
        if isinstance(other, SchemaNode):
            if isinstance(other, AtomicNode):
                return other in self.constituents
            elif isinstance(other, ProductNode):
                from schema.helpers.is_sublist import is_sublist
                return is_sublist(other.constituents, self.constituents) and self != other
            return False
        raise NotImplemented()

    def __ge__(self, other):
        if isinstance(other, SchemaNode):
            if isinstance(other, AtomicNode):
                return other in self.constituents
            elif isinstance(other, ProductNode):
                from schema.helpers.is_sublist import is_sublist
                return is_sublist(other.constituents, self.constituents)
            return False
        raise NotImplemented()

    def __repr__(self):
        return ";".join([str(c) for c in self.constituents])

    def __str__(self):
        return self.__repr__()


class SchemaClass(SchemaNode):

    def __init__(self, name):
        self.name = name

    def __hash__(self):
        return hash(self.name)

    def __eq__(self, other):
        if isinstance(other, AtomicNode):
            return False
        elif isinstance(other, ProductNode):
            return False
        elif isinstance(other, SchemaClass):
            return self.name == other.name
        else:
            raise NotImplemented()

    def __le__(self, other):
        if isinstance(other, SchemaNode):
            return False
        raise NotImplemented()

    def __lt__(self, other):
        if isinstance(other, SchemaNode):
            return False
        raise NotImplemented()

    def __ge__(self, other):
        if isinstance(other, SchemaNode):
            return False
        raise NotImplemented()

    def __gt__(self, other):
        if isinstance(other, SchemaNode):
            return False
        raise NotImplemented()

    def __repr__(self):
        return f"{self.name} <Class>"

    def __str__(self):
        return self.name