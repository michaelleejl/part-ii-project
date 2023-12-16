import pandas as pd

from backend.pandas_backend.exceptions import KeyDuplicationException
from backend.pandas_backend.pandas_backend import PandasBackend
from schema.edge import SchemaEdge
from schema.exceptions import ClusterAlreadyExistsException, NodesDoNotExistInGraphException, \
    ClassAlreadyExistsException, CannotRenameClassException
from schema.graph import SchemaGraph
from schema.node import SchemaNode
from tables.derivation import Get, End, StartTraversal, EndTraversal
from tables.function import Function
from tables.raw_column import RawColumn
from tables.table import Table


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
        self.clusters = frozenset()
        self.backend = None

    def insert_dataframe(self, df, cluster):
        if self.backend is None:
            self.backend = PandasBackend()
        else:
            assert type(self.backend) == PandasBackend

        if cluster in self.clusters:
            raise ClusterAlreadyExistsException(cluster)
        else:
            self.clusters = self.clusters.union(frozenset([cluster]))

        keys = df.index.to_frame().reset_index(drop=True)
        check_for_duplicate_keys(keys)

        key_names = keys.columns.to_list()
        val_names = df.columns.to_list()

        key_nodes = [SchemaNode(name.lower(), cluster=cluster) for name in key_names]
        val_nodes = [SchemaNode(name.lower(), cluster=cluster) for name in val_names]

        key_node = SchemaNode.product(key_nodes)
        nodes = key_nodes + val_nodes

        dfa = df.reset_index()

        for node in nodes:
            self.backend.map_atomic_node_to_domain(node, pd.DataFrame(dfa[node.name]).drop_duplicates())

        # for node in key_names:
        #     self.backend.map_edge_to_data_relation(SchemaEdge(key_node, SchemaNode(node, cluster=cluster)), dfa[key_names].drop_duplicates())

        for node in val_names:
            self.backend.map_edge_to_data_relation(SchemaEdge(key_node, SchemaNode(node, cluster=cluster)), df[node].reset_index().drop_duplicates())

        self.schema_graph.add_nodes(nodes)
        self.schema_graph.add_cluster(nodes, key_node)

    def add_node(self, name):
        node = self.schema_graph.get_node_not_in_graph_with_name(name.lower())
        self.schema_graph.add_node(node)
        return node

    def add_edge(self, node1, node2, cardinality):
        edge = SchemaEdge(node1, node2, cardinality)
        self.schema_graph.add_edge(node1, node2, cardinality)
        return edge

    def map_edge_to_closure_function(self, edge, function: Function):
        self.backend.map_edge_to_closure_function(edge, function)

    def blend(self, node1: SchemaNode, node2: SchemaNode, under: str = None):
        if under is not None:
            classname = SchemaNode(under, cluster="type")
            clss1 = self.schema_graph.equivalence_class.get_classname(node1)
            clss2 = self.schema_graph.equivalence_class.get_classname(node2)
            if classname not in self.schema_graph.schema_nodes and clss1 is None and clss2 is None:
                domain = pd.DataFrame([], columns=[under])
                self.backend.map_atomic_node_to_domain(classname, domain)
                self.schema_graph.add_class(classname)
            elif classname in self.schema_graph.schema_nodes and (clss1 is None and clss2 is None):
                raise ClassAlreadyExistsException()
            elif classname in self.schema_graph.schema_nodes and (clss1 is not None and clss1 != classname) and (clss2 is not None and clss2 != classname):
                raise CannotRenameClassException()
            new_members = frozenset()
            if clss1 is None:
                new_members = new_members.union(self.schema_graph.equivalence_class.attach_classname(node1, classname))
            if clss2 is None:
                new_members = new_members.union(self.schema_graph.equivalence_class.attach_classname(node2, classname))
            self.schema_graph.blend_nodes(node1, node2)
            self.schema_graph.blend_nodes(node1, classname)
            if new_members is not None:
                for member in new_members:
                    self.backend.extend_domain(classname, member)
        else:
            self.schema_graph.blend_nodes(node1, node2)

    def clone(self, node: SchemaNode, name: str):
        assert len(SchemaNode.get_constituents(node)) == 1
        new_node = SchemaNode(name, cluster=node.cluster)
        if new_node in self.schema_graph.schema_nodes:
            assert self.schema_graph.are_nodes_equal(new_node, node)
            return new_node
        self.schema_graph.add_node(new_node)
        self.backend.clone(node, new_node)
        self.schema_graph.blend_nodes(new_node, node)
        return new_node

    def get(self, keys: list[str], with_names: list[str] = None):
        names = keys
        keys = [self.schema_graph.get_node_in_graph_with_name(k) for k in keys]
        key_set = frozenset(keys)
        diff = key_set.difference(self.schema_graph.schema_nodes)
        if len(diff) > 0:
            raise NodesDoNotExistInGraphException(list(diff))
        # if with_names is not None:
        #     assert len(with_names) == len(keys)
        #     keys = [self.clone(k, with_names[i]) if with_names[i] is not None else k for i, k in enumerate(keys)]
        if with_names is None:
            with_names = names
        key_node = SchemaNode.product(keys)
        key_nodes = SchemaNode.get_constituents(key_node)
        columns = list(zip(key_nodes, with_names))
        return Table.construct(columns, self)

    def get_node_with_name(self, name: str) -> SchemaNode:
        return self.schema_graph.get_node_in_graph_with_name(name)

    def find_shortest_path(self, node1: SchemaNode, node2: SchemaNode, via: list[SchemaNode] = None, backwards=False):
        return self.schema_graph.find_shortest_path(node1, node2, via, backwards)

    def find_shortest_path_between_columns(self, from_columns: list[RawColumn], to_columns: list[RawColumn], explicit_keys, via: list[SchemaNode] = None, backwards=False):
        node1 = SchemaNode.product([c.node for c in from_columns])
        node2 = SchemaNode.product([c.node for c in to_columns])
        path, commands, hidden_keys = self.find_shortest_path(node1, node2, via, backwards)
        first = StartTraversal(from_columns, explicit_keys)
        last = EndTraversal(from_columns, to_columns)
        if len(commands) > 0:
            return path, [first] + commands + [last], hidden_keys
        else:
            return path, [first, last], hidden_keys

    def execute_query(self, table_id, derived_from, derivation):
        x, y, z, new_backend = self.backend.execute_query(table_id, derived_from, derivation)
        self.backend = new_backend
        return x, y, z, self

    def __repr__(self):
        return self.schema_graph.__repr__()

    def __str__(self):
        return self.__repr__()