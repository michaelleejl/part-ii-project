import pandas as pd

from schema.edge import SchemaEdge
from schema.equality import SchemaEquality
from schema.exceptions import FamilyAlreadyExistsException
from schema.graph import SchemaGraph
from schema.helpers.check_for_duplicate_keys import check_for_duplicate_keys
from schema.node import SchemaNode
from schema.type import SchemaType
from union_find.union_find import UnionFindItem


class Schema:
    def __init__(self):
        self.schema_graph = SchemaGraph()
        self.schema_types = {}
        self.families = frozenset()

    def insert(self, data_frame, family):
        if family in self.families:
            raise FamilyAlreadyExistsException(family)
        else:
            self.families = self.families.union(frozenset([family]))
        df = pd.DataFrame.copy(data_frame)
        keys = df.index.to_frame().reset_index(drop=True)
        check_for_duplicate_keys(keys)

        key_names = keys.columns.to_list()
        val_names = df.columns.to_list()
        key_node = SchemaNode(', '.join(key_names), keys, family)

        def create_nodes_and_mappings(v, tbl):
            n = SchemaNode(v, tbl[v].drop_duplicates(), family)
            is_v_a_key = v in set(key_names)
            if is_v_a_key:
                project = key_names
            else:
                project = key_names + [v]
            m = tbl.reset_index()[project].drop_duplicates()

            val_col_name = n.get_unique_col_name(v)
            if is_v_a_key:
                m[val_col_name] = m[v]
                m = m.rename({k: key_node.get_unique_col_name(k) for k in key_names}, axis=1)
            else:
                m = m.rename({k: key_node.get_unique_col_name(k) for k in key_names}, axis=1)
                m = m.rename({v: val_col_name}, axis=1)
            return n, m

        key_values = [create_nodes_and_mappings(k, keys) for k in key_names]
        values = [create_nodes_and_mappings(v, df) for v in val_names]

        v_nodes = []
        for (val_node, mapping) in key_values + values:
            self.schema_graph.add_relation(SchemaEdge(key_node, val_node, mapping))
            v_nodes += [val_node]

        self.schema_graph.add_nodes(frozenset(v_nodes + [key_node]))

    def get_node(self, name: str, family: str):
        return self.schema_graph.get_node(name, family)

    def blend(self, node1: SchemaNode, node2: SchemaNode, under_type: str, with_equivalence):
        name = under_type
        equivalence_relation = with_equivalence

        if name not in self.schema_types:
            self.schema_types[name] = SchemaType.construct(name, node1, node2, equivalence_relation)
        else:
            self.schema_types[name] = SchemaType.update(self.schema_types[name], node1, node2, equivalence_relation)

        schema_type = self.schema_types[name]

        edge1 = node1.data
        edge1[f"{hash(schema_type)}_class"] = pd.Series(node1.get_values()).apply(lambda v: schema_type.get_class(UnionFindItem(v, node1)))

        edge2 = node2.data
        edge2[f"{hash(schema_type)}_class"] = pd.Series(node2.get_values()).apply(lambda v: schema_type.get_class(UnionFindItem(v, node2)))

        if self.schema_graph.does_relation_exist(schema_type, node1):
            e1 = self.schema_graph.get_edge_between_nodes(schema_type, node1)
            self.schema_graph.extend_relation(e1, edge1)
        else:
            self.schema_graph.add_relation(SchemaEquality(schema_type, node1, edge1))
        if self.schema_graph.does_relation_exist(schema_type, node2):
            e2 = self.schema_graph.get_edge_between_nodes(schema_type, node2)
            self.schema_graph.extend_relation(e2, edge2)
        else:
            self.schema_graph.add_relation(SchemaEquality(schema_type, node2, edge2))

    def __repr__(self):
        return self.schema_graph.__repr__()

    def __str__(self):
        return self.__repr__()