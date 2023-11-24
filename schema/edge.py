import numpy as np
import pandas as pd

from schema.cardinality import Cardinality
from schema.exceptions import ExtendMappingShouldConflictWithOldMapping, EdgeIsNotConnectedToNodeException
from schema.helpers.determine_cardinality import determine_cardinality
from schema.helpers.invert_cardinality import invert_cardinality
from schema.node import SchemaNode


class SchemaEdge:
    def __init__(self, from_node: SchemaNode, to_node: SchemaNode, mapping):
        self.from_node = from_node
        self.to_node = to_node
        columns = mapping.columns
        from_columns = list(filter(lambda c: c.startswith(str(hash(from_node))), columns))
        to_columns = list(filter(lambda c: c.startswith(str(hash(to_node))), columns))
        self.cardinality = determine_cardinality(mapping, from_columns, to_columns)
        self.mapping = mapping

    def get_cardinality(self, from_node: SchemaNode) -> Cardinality:
        if from_node == self.from_node:
            return self.cardinality
        elif from_node == self.to_node:
            return invert_cardinality(self.cardinality)
        else:
            raise EdgeIsNotConnectedToNodeException(self, from_node)

    @classmethod
    def extend_mapping(cls, relation, additional_mapping):
        assert np.all(additional_mapping.columns == relation.mapping.columns)
        from_node_columns = list(relation.from_node.data.columns)
        to_node_columns = list(relation.to_node.data.columns)

        relation_map = relation.mapping.astype(object)
        additional_mapping = additional_mapping.astype(object)

        merged = relation_map.merge(additional_mapping.astype(object), on=from_node_columns, indicator=True, how='outer').astype(object)
        intersect = merged[merged['_merge'] == 'both']
        for to_node_col in to_node_columns:
            if not np.all(intersect[f"{to_node_col}_x"] == intersect[f"{to_node_col}_y"]):
                raise ExtendMappingShouldConflictWithOldMapping(intersect)
        updated_map = merged[(merged['_merge'] == 'right_only')][from_node_columns + [f"{col}_y" for col in to_node_columns]].astype(object)
        renaming = {f"{c}_y": c for c in to_node_columns}
        updated_map = updated_map.rename(renaming, axis=1)
        new_mapping = pd.DataFrame(pd.concat([relation_map, updated_map], ignore_index=True))
        return SchemaEdge(relation.from_node, relation.to_node, new_mapping)

    @classmethod
    def update_mapping(cls, relation, additional_mapping):
        assert np.all(additional_mapping.columns == relation.mapping.columns)
        from_node_columns = list(relation.from_node.data.columns)
        to_node_columns = list(relation.to_node.data.columns)

        relation_map = relation.mapping.copy().astype(object)
        additional_mapping = additional_mapping.copy().astype(object)

        merged = relation_map.merge(additional_mapping.astype(object), on=from_node_columns, indicator=True,
                                        how='outer').astype(object)
        for col in to_node_columns:
            left = f"{col}_x"
            right = f"{col}_y"
            merged[col] = merged.apply(lambda r: r[right] if r["_merge"] != "left_only" else r[left], axis=1)

        updated_map = merged[from_node_columns + to_node_columns].astype(object)
        return SchemaEdge(relation.from_node, relation.to_node, updated_map)

    def __key(self):
        return self.from_node, self.to_node

    def __hash__(self):
        return hash(self.__key())

    def __eq__(self, other):
        if isinstance(other, SchemaEdge):
            return self.__key() == other.__key()
        return NotImplemented

    def __repr__(self):
        return self.__str__() + f"\nmapping has {len(self.mapping)} entries "

    def __str__(self):
        if self.cardinality == Cardinality.ONE_TO_ONE:
            arrow = "<-->"
        elif self.cardinality == Cardinality.MANY_TO_ONE:
            arrow = "--->"
        elif self.cardinality == Cardinality.ONE_TO_MANY:
            arrow = "<---"
        else:
            arrow = "---"
        return f"{self.from_node.name} [{self.from_node.family}] {arrow} {self.to_node.name} [{self.to_node.family}]"
