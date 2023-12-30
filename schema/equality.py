from schema import Cardinality
from schema.edge import SchemaEdge
from schema.node import SchemaNode


class SchemaEquality(SchemaEdge):
    def __init__(self, from_node: SchemaNode, to_node: SchemaNode):
        super().__init__(from_node, to_node, Cardinality.ONE_TO_ONE)

    def __repr__(self):
        arrow = "==="
        return f"{self.from_node} {arrow} {self.to_node}"

    def is_equality(self):
        return True

    def __str__(self):
        return self.__repr__()

    def is_equality(self):
        return True

