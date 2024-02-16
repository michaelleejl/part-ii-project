from schema.cardinality import Cardinality
from schema.edge import SchemaEdge
from schema.node import SchemaNode


class SchemaEquality(SchemaEdge):
    """
    SchemaEquality represents an equality edge in a schema graph
    An equality edge is an edge that represents that two nodes are in the same equivalence class
    """

    def __init__(self, from_node: SchemaNode, to_node: SchemaNode):
        super().__init__(from_node, to_node, Cardinality.ONE_TO_ONE)

    def __repr__(self):
        arrow = "==="
        return f"{self.from_node} {arrow} {self.to_node}"

    def is_equality(self):
        """
        Returns whether the edge is an equality edge

        Returns:
            bool: True
        """
        return True

    def __str__(self):
        return self.__repr__()
