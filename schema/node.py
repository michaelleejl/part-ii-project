from __future__ import annotations

import abc
import uuid

import numpy as np

from schema.base_types import BaseType
from union_find.union_find import UnionFind


class SchemaNode(abc.ABC):
    """SchemaNode is an abstract class that represents a node in a schema graph"""

    @classmethod
    def product(cls, nodes: list[SchemaNode]) -> ProductNode:
        """Takes a list of nodes and returns their product

        Args:
            nodes (list[SchemaNode]): list of nodes

        Returns:
            ProductNode: the node formed by taking the product of the nodes in the list
        """

        atomics = []
        for node in nodes:
            cs = SchemaNode.get_constituents(node)
            atomics += cs
        constituents = atomics
        if len(constituents) == 1:
            (node,) = atomics
            return node
        else:
            return ProductNode(constituents)

    @classmethod
    def get_constituents(cls, node: SchemaNode) -> list[AtomicNode | SchemaClass]:
        """
        Returns the constituents of a node. If the node is atomic, returns the node itself.
        If the node is a product, returns its constituent atomic nodes

        Args:
            node (SchemaNode): The node whose constituents are to be found

        Returns:
            list[SchemaNode]: The constituents of the node
        """
        if isinstance(node, AtomicNode) or isinstance(node, SchemaClass):
            return [node]
        else:
            assert isinstance(node, ProductNode)
            return node.constituents

    @classmethod
    def is_atomic(cls, node: SchemaNode) -> bool:
        """
        Returns True if the node is atomic, False otherwise

        Args:
            node (SchemaNode): The node to check

        Returns:
            bool: True if the node is atomic, False otherwise
        """

        return isinstance(node, AtomicNode)

    @classmethod
    def is_equivalent(
        cls, node1: SchemaNode, node2: SchemaNode, equivalence_class: UnionFind
    ) -> bool:
        """
        Returns True if the two nodes are equivalent under the given equivalence class, False otherwise

        Args:
            node1 (SchemaNode): The first node
            node2 (SchemaNode): The second node
            equivalence_class (UnionFind): The equivalence class

        Returns:
            bool: True if the two nodes are equivalent under the given equivalence class, False otherwise
        """
        c1 = SchemaNode.get_constituents(node1)
        c2 = SchemaNode.get_constituents(node2)
        if len(c1) != len(c2):
            return False
        eq = np.all(
            np.array(
                list(
                    map(
                        lambda x: equivalence_class.find_leader(x[0])
                        == equivalence_class.find_leader(x[1]),
                        zip(c1, c2),
                    )
                )
            )
        )
        return eq

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


class AtomicNode(SchemaNode):
    """
    AtomicNode is a class that represents an atomic node in a schema graph
    That is, it represents a node that cannot be represented as a product of other nodes
    """

    def __init__(self, name: str, node_type: BaseType = BaseType.OBJECT):
        """
        Initializes an AtomicNode

        Args:
            name (str): The name of the node
            node_type (BaseType): The type of the node

        Returns:
            AtomicNode: The initialized AtomicNode
        """
        super().__init__()
        self.node_type = node_type
        self.name = name
        self.id = uuid.uuid4()
        self.id_prefix = 3

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
        if self.id_prefix == 0:
            return self.name
        else:
            return f"{self.name} <{str(self.id)[:self.id_prefix]}>"

    def __str__(self):
        return self.__repr__()


class ProductNodeShouldHaveAtLeastTwoConstituentsException(Exception):
    def __init__(self, constituents):
        super().__init__(
            f"Product node should have at least two constituents. Constituents: {constituents}"
        )


class ProductNode(SchemaNode):
    """
    ProductNode is a class that represents a node formed by taking the product of some number of atomic nodes
    """

    def __init__(self, constituents: list[AtomicNode]):
        """
        Initializes a ProductNode

        Args:
            constituents (list[AtomicNode]): The atomic nodes that form the product
        """
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

                return (
                    is_sublist(self.constituents, other.constituents) and self != other
                )
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

                return (
                    is_sublist(other.constituents, self.constituents) and self != other
                )
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
        return ";".join([repr(c) for c in self.constituents])

    def __str__(self):
        return self.__repr__()


class SchemaClass(SchemaNode):
    """SchemaClass is a class that represents a class node in a schema graph"""

    def __init__(self, name: str):
        """Initializes a SchemaClass

        Args:
            name (str): The name of the class
        """
        self.name = name
        self.node_type = None

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
