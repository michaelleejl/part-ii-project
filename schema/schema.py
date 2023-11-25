import pandas as pd

from backend.pandas_backend.exceptions import KeyDuplicationException
from backend.pandas_backend.relation import DataRelation
from schema import Cardinality
from schema.edge import SchemaEdge
from schema.equality import SchemaEquality
from schema.exceptions import FamilyAlreadyExistsException
from schema.graph import SchemaGraph
from schema.node import SchemaNode
from schema.type import SchemaType
from union_find.union_find import UnionFindItem
from backend.pandas_backend.pandas_backend import PandasBackend


def check_for_duplicate_keys(keys):
    duplicates = keys[keys.duplicated()].drop_duplicates()
    if len(keys.columns) > 1:
        duplicates = duplicates.itertuples(index=False, name=None)
    else:
        duplicates = duplicates.values
    duplicates_str = [str(k) for k in duplicates]
    if len(duplicates_str) > 0:
        raise KeyDuplicationException('\n'.join(duplicates_str))


class Schema:
    def __init__(self):
        self.schema_graph = SchemaGraph()
        self.schema_types = {}
        self.families = frozenset()
        self.backend = None

    def insert_dataframe(self, df, family):
        if self.backend is None:
            self.backend = PandasBackend()
        else:
            assert type(self.backend) == PandasBackend

        if family in self.families:
            raise FamilyAlreadyExistsException(family)
        else:
            self.families = self.families.union(frozenset([family]))

        keys = df.index.to_frame().reset_index(drop=True)
        check_for_duplicate_keys(keys)

        key_names = keys.columns.to_list()
        val_names = df.columns.to_list()
        key_node = SchemaNode(', '.join(key_names), family)
        self.backend.map_node_to_domain(key_node, keys)

        def create_nodes_and_mappings(v, tbl):
            n = SchemaNode(v, family)
            is_v_a_key = v in set(key_names)
            if is_v_a_key:
                project = key_names
            else:
                project = key_names + [v]
            m = tbl.reset_index()[project].drop_duplicates()
            val_col_name = n.prepend_id(v)

            if is_v_a_key:
                m[val_col_name] = m[v]
                m = m.rename({k: key_node.prepend_id(k) for k in key_names}, axis=1)
            else:
                m = m.rename({k: key_node.prepend_id(k) for k in key_names}, axis=1)
                m = m.rename({v: val_col_name}, axis=1)
            return n, m

        key_values = [create_nodes_and_mappings(k, keys) for k in key_names]
        values = [create_nodes_and_mappings(v, df) for v in val_names]

        v_nodes = []
        for (val_node, mapping) in key_values + values:
            edge = SchemaEdge(key_node, val_node, Cardinality.MANY_TO_MANY)
            self.backend.map_edge_to_relation(edge, DataRelation(mapping))
            cardinality = self.backend.get_cardinality(edge, key_node)
            self.schema_graph.add_edge(SchemaEdge(key_node, val_node, cardinality))
            v_nodes += [val_node]

        self.schema_graph.add_nodes(frozenset(v_nodes + [key_node]))

    def get_node(self, name: str, family: str):
        return self.schema_graph.get_node(name, family)

    def blend(self, node1: SchemaNode, node2: SchemaNode, under_type: str):
        name = under_type

        if name not in self.schema_types:
            self.schema_types[name] = SchemaType.construct(name, node1, node2)
        else:
            self.schema_types[name] = SchemaType.update(self.schema_types[name], [node1, node2])
        #
        # schema_type = self.schema_types[name]
        #
        # edge1 = node1.data
        # edge1[f"{hash(schema_type)}_class"] = pd.Series(node1.get_values()).apply(lambda v: schema_type.get_class(UnionFindItem(v, node1)))
        #
        # edge2 = node2.data
        # edge2[f"{hash(schema_type)}_class"] = pd.Series(node2.get_values()).apply(lambda v: schema_type.get_class(UnionFindItem(v, node2)))
        #
        # if self.schema_graph.does_relation_exist(schema_type, node1):
        #     e1 = self.schema_graph.get_edge_between_nodes(schema_type, node1)
        #     self.schema_graph.extend_relation(e1, edge1)
        # else:
        #     self.schema_graph.add_edge(SchemaEquality(schema_type, node1, edge1))
        # if self.schema_graph.does_relation_exist(schema_type, node2):
        #     e2 = self.schema_graph.get_edge_between_nodes(schema_type, node2)
        #     self.schema_graph.extend_relation(e2, edge2)
        # else:
        #     self.schema_graph.add_edge(SchemaEquality(schema_type, node2, edge2))

    def clone(self, node: SchemaNode, name=None):
        if name is None:
            i = 1
            name = f"{node.name} {i}"
            candidate = SchemaNode(name, node.family)
            while candidate in self.schema_graph:
                i += 1
                name = f"{node.name} {i}"
            if node.clone_type is None:
                clone_type = SchemaType(node.name, frozenset([node, candidate]))
                node.clone_type = clone_type
            else:
                clone_type = node.clone_type
            candidate = SchemaNode(name, node.family, clone_type)
            self.blend(candidate, node, under_type=clone_type.name)

    def __repr__(self):
        return self.schema_graph.__repr__()

    def __str__(self):
        return self.__repr__()