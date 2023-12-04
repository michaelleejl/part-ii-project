from schema.edge import SchemaEdge
from schema.node import SchemaNode


class SchemaEquality(SchemaEdge):
    def __init__(self, from_node: SchemaNode, to_node: SchemaNode, mapping):
        super().__init__(from_node, to_node, mapping)

    def __repr__(self):
        arrow = "==="
        return f"{self.from_node.name} [{self.from_node.family}] {arrow} {self.to_node.name} [{self.to_node.family}]"

    def __str__(self):
        return self.__repr__()

    def is_equality(self):
        return True

