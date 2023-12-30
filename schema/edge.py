from enum import Enum

from backend.backend import Backend
from schema.cardinality import Cardinality
from schema.node import SchemaNode


def reverse_cardinality(cardinality: Cardinality):
    match cardinality:
        case Cardinality.ONE_TO_MANY:
            return Cardinality.MANY_TO_ONE
        case Cardinality.MANY_TO_ONE:
            return Cardinality.ONE_TO_MANY
        case _:
            return cardinality


class SchemaEdge:
    def __init__(self, from_node: SchemaNode, to_node: SchemaNode, cardinality: Cardinality = Cardinality.MANY_TO_MANY):
        self.from_node = from_node
        self.to_node = to_node
        self.cardinality = cardinality

    def __key(self):
        return self.from_node, self.to_node

    def traverse(self, start):
        if start == self.from_node:
            return self.to_node
        else:
            return self.from_node

    def is_equality(self):
        return False

    def get_cardinality(self, starting_from: SchemaNode):
        if starting_from == self.from_node:
            return self.cardinality
        elif starting_from == self.to_node:
            return reverse_cardinality(self.cardinality)

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
