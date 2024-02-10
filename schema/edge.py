from __future__ import annotations

from schema.cardinality import Cardinality
from schema.node import SchemaNode


def reverse_cardinality(cardinality: Cardinality) -> Cardinality:
    """
    If X->Y has cardinality c, then Y->X has cardinality invert_cardinality(c)

    Args:
        cardinality (Cardinality): The cardinality of the edge to be inverted

    Returns:
        Cardinality: The cardinality of the inverted edge
    """
    match cardinality:
        case Cardinality.ONE_TO_MANY:
            return Cardinality.MANY_TO_ONE
        case Cardinality.MANY_TO_ONE:
            return Cardinality.ONE_TO_MANY
        case _:
            return cardinality


class SchemaEdge:
    """
    SchemaEdge represents an edge in a schema graph
    """

    def __init__(
        self,
        from_node: SchemaNode,
        to_node: SchemaNode,
        cardinality: Cardinality = Cardinality.MANY_TO_MANY,
    ):
        self.from_node: SchemaNode = from_node
        self.to_node: SchemaNode = to_node
        self.cardinality: Cardinality = cardinality

    def __key(self):
        return self.from_node, self.to_node

    def traverse(self, start: SchemaNode) -> SchemaNode:
        """
        Traverses the edge from the start node to the end node

        Args:
            start (SchemaNode): The start node

        Returns:
            SchemaNode: The end node
        """
        if start == self.from_node:
            return self.to_node
        else:
            return self.from_node

    def is_equality(self) -> bool:
        """
        Returns whether the edge is an equality edge

        Returns:
            bool: False
        """
        return False

    def get_cardinality(self, starting_from: SchemaNode) -> Cardinality:
        """
        Returns the cardinality of the edge when traversing from the given node

        Args:
            starting_from (SchemaNode): The node to start traversing from

        Returns:
            Cardinality: The cardinality of the edge when traversing from the given node
        """
        if starting_from == self.from_node:
            return self.cardinality
        elif starting_from == self.to_node:
            return reverse_cardinality(self.cardinality)

    def is_functional(self) -> bool:
        """
        Returns whether the edge is functional

        Returns:
            bool: True if the edge is functional, False otherwise
        """
        return (
            self.cardinality == Cardinality.ONE_TO_ONE
            or self.cardinality == Cardinality.MANY_TO_ONE
        )

    def get_hidden_keys(self) -> list[SchemaNode]:
        """
        Returns the hidden keys needed to traverse the edge

        Returns:
            list[SchemaNode]: The hidden keys needed to traverse the edge
        """
        end = self.to_node
        if not self.is_functional():
            return SchemaNode.get_constituents(end)
        else:
            return []

    @classmethod
    def invert(cls, edge: SchemaEdge) -> SchemaEdge:
        """
        Inverts the edge

        Args:
            edge (SchemaEdge): The edge to be inverted

        Returns:
            SchemaEdge: The inverted edge
        """
        return SchemaEdge(
            edge.to_node, edge.from_node, reverse_cardinality(edge.cardinality)
        )

    def __hash__(self):
        return hash(self.__key())

    def __eq__(self, other):
        if isinstance(other, SchemaEdge):
            return self.__key() == other.__key()
        return NotImplemented

    def __repr__(self):
        return self.__str__()

    def __str__(self):
        if self.cardinality == Cardinality.ONE_TO_ONE:
            arrow = "<-->"
        elif self.cardinality == Cardinality.MANY_TO_ONE:
            arrow = "--->"
        elif self.cardinality == Cardinality.ONE_TO_MANY:
            arrow = "<---"
        else:
            arrow = "---"
        return f"{self.from_node} {arrow} {self.to_node}"
