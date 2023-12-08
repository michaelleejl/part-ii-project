from backend.pandas_backend.exceptions import KeyDuplicationException
from backend.pandas_backend.pandas_backend import PandasBackend
from schema.exceptions import ClusterAlreadyExistsException, NodesDoNotExistInGraphException
from schema.graph import SchemaGraph
from schema.node import SchemaNode
from tables.derivation import DerivationNode
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

        key_nodes = [SchemaNode(name, cluster=cluster) for name in key_names]
        val_nodes = [SchemaNode(name, cluster=cluster) for name in val_names]

        key_node = SchemaNode.product(key_nodes)
        nodes = key_nodes + val_nodes
        self.schema_graph.add_nodes(nodes)
        self.schema_graph.add_cluster(nodes, key_node)

    def add_node(self, name, cluster):
        self.schema_graph.add_node(SchemaNode(name, cluster=cluster))

    def blend(self, node1: SchemaNode, node2: SchemaNode, under: str = None):
        self.schema_graph.blend_nodes(node1, node2, under)

    def clone(self, node: SchemaNode, name: str = None):
        i = 1
        if name is None:
            name = node.name
            candidate = SchemaNode(f"{node.name}_{i}", cluster=node.cluster)
        else:
            candidate = SchemaNode(name, cluster=node.cluster)
            i = 0
        while candidate in self.schema_graph.schema_nodes:
            i += 1
            candidate = SchemaNode(f"{name}_{i}", cluster=node.cluster)

        self.schema_graph.add_node(candidate)
        self.blend(candidate, node)
        return candidate

    def get(self, keys: list[str]):
        keys = [self.schema_graph.get_node_with_name(k) for k in keys]
        key_set = frozenset(keys)
        diff = key_set.difference(self.schema_graph.schema_nodes)
        if len(diff) > 0:
            raise NodesDoNotExistInGraphException(list(diff))
        key_node = SchemaNode.product(keys)
        root = key_node
        key_nodes = SchemaNode.get_constituents(key_node)
        derivation = DerivationNode(root, frozenset([]))
        return Table.construct(key_nodes, derivation, self)

    def get_node_with_name(self, name: str) -> SchemaNode:
        return self.schema_graph.get_node_with_name(name)

    def find_shortest_path(self, node1: SchemaNode, node2: SchemaNode, via: list[SchemaNode] = None):
        self.schema_graph.find_shortest_path(node1, node2, via)

    def __repr__(self):
        return self.schema_graph.__repr__()

    def __str__(self):
        return self.__repr__()