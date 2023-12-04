import numpy as np
import pandas as pd

from schema.node import SchemaNode
from union_find.union_find import UnionFind, UnionFindItem


class SchemaClass(SchemaNode):
    def __init__(self, name, nodes):
        self.nodes = nodes
        super().__init__(name, "types")

    @classmethod
    def construct(cls, name, node1: SchemaNode, node2: SchemaNode):
        return SchemaClass(name, frozenset([node1, node2]))

    @classmethod
    def update(cls, schema_type, nodes: list[SchemaNode]):
        return SchemaClass(schema_type.name, schema_type.nodes.union(nodes))

    def __contains__(self, node):
        return node in self.nodes