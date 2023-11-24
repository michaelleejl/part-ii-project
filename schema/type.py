import numpy as np
import pandas as pd

from schema.node import SchemaNode
from union_find.union_find import UnionFind, UnionFindItem


class SchemaType(SchemaNode):
    def __init__(self, name, classes, uf: UnionFind, nodes):
        self.uf = uf
        self.nodes = nodes
        self.classes = classes
        df = {"class": list(classes)}
        super().__init__(name, df, "types")

    @classmethod
    def construct(cls, name, node1: SchemaNode, node2: SchemaNode, equivalence_relation: pd.DataFrame):
        uf = UnionFind.initialise()
        vals = frozenset()
        for node in [node1, node2]:
            vs = node.get_values()
            uf = UnionFind.add_singletons(uf, vs, node)
            vals = vals.union([UnionFindItem(v, node) for v in vs])
        for (x, y) in zip(equivalence_relation[node1], equivalence_relation[node2]):
            uf = UnionFind.union(uf, UnionFindItem(x, node1), UnionFindItem(y, node2))
        classes = frozenset()
        for v in vals:
            classes.union([uf.find_leader(v)])
        return SchemaType(name, classes, uf, frozenset([node1, node2]))

    @classmethod
    def update(cls, schema_type, node1: SchemaNode, node2: SchemaNode, equivalence_relation: pd.DataFrame):
        uf = schema_type.uf
        vals = frozenset()
        for node in [node1, node2]:
            vs = node.get_values()
            uf = UnionFind.add_singletons(uf, vs, node)
            vals = vals.union([UnionFindItem(v, node) for v in vs])
        for (x, y) in zip(equivalence_relation[node1], equivalence_relation[node2]):
            uf = UnionFind.union(uf, UnionFindItem(x, node1), UnionFindItem(y, node2))
        classes = frozenset()
        for v in vals:
            classes = classes.union([uf.find_leader(v)])
        new_classes = schema_type.classes.union(classes)
        return SchemaType(schema_type.name, new_classes, uf, [node1, node2])

    def get_class(self, item: UnionFindItem) -> np.nan | UnionFindItem:
        if item in self.uf:
            return self.uf.find_leader(item)
        else:
            return np.nan

    def __contains__(self, node):
        return node in self.nodes