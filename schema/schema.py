import pandas as pd

from backend.pandas_backend.determine_base_type_of_columns import (
    determine_base_type_of_columns,
)
from backend.pandas_backend.exceptions import KeyDuplicationException
from backend.pandas_backend.pandas_backend import PandasBackend
from exp.exp import Exp
from schema.cardinality import Cardinality
from schema.edge import SchemaEdge
from schema.exceptions import (
    NodesDoNotExistInGraphException,
    ClassAlreadyExistsException,
    CannotInsertDataFrameIfSchemaBackedBySQLBackendException,
    SchemaClassMustBeSpecifiedException,
    CannotBlendNodesUnderDifferentClassesException,
    CannotBlendNodesWithDifferentTypeException,
    ColumnMustBeAnAtomicNodeOrClassException,
)
from schema.graph import SchemaGraph
from schema.node import SchemaNode, AtomicNode, SchemaClass
from representation.representation import StartTraversal, EndTraversal
from frontend.domain import Domain
from frontend.tables.table import Table


def check_for_duplicate_keys(keys):
    duplicates = keys[keys.duplicated()].drop_duplicates()
    if len(keys.columns) > 1:
        duplicates = duplicates.itertuples(index=False, name=None)
    else:
        duplicates = duplicates.values
    duplicates_str = [str(k) for k in duplicates]
    if len(duplicates_str) > 0:
        raise KeyDuplicationException("\n".join(duplicates_str))


class Schema:
    """A Schema holds a SchemaGraph and a Backend"""

    def __init__(self):
        """Creates a new Schema with an empty graph and no backend"""
        self.schema_graph = SchemaGraph()
        self.backend = None

    def insert_dataframe(self, df: pd.DataFrame) -> dict[str, AtomicNode]:
        """Inserts a dataframe with non-empty index into the Schema.
        The backend must be of type PandasBackend.
        If the Backend is None, it is automatically initialised as a new PandasBackend.
        Each column is a new node in the schema.
        An edge between the indices and each value column is added.

        Args:
            df (pd.DataFrame): A pandas DataFrame with non-empty index.

        Returns:
            dict[str, AtomicNode]: A dictionary from name of column in the df to corresponding node in the SchemaGraph
        """
        if self.backend is None:
            self.backend = PandasBackend()
        else:
            if not isinstance(self.backend, PandasBackend):
                raise CannotInsertDataFrameIfSchemaBackedBySQLBackendException()

        keys = df.index.to_frame().reset_index(drop=True)
        check_for_duplicate_keys(keys)

        key_names = keys.columns.to_list()
        key_types = determine_base_type_of_columns(keys)
        val_names = df.columns.to_list()
        val_types = determine_base_type_of_columns(df)

        key_nodes = [
            AtomicNode(name.lower(), node_type)
            for (name, node_type) in zip(key_names, key_types)
        ]
        val_nodes = [
            AtomicNode(name.lower(), node_type)
            for (name, node_type) in zip(val_names, val_types)
        ]

        key_node = SchemaNode.product(key_nodes)
        nodes = key_nodes + val_nodes

        dfa = df.reset_index()

        for node in nodes:
            self.backend.map_atomic_node_to_domain(
                node, pd.DataFrame(dfa[node.name]).drop_duplicates()
            )

        for node in val_nodes:
            self.backend.map_edge_to_data_relation(
                SchemaEdge(key_node, node),
                df[node.name].reset_index().drop_duplicates(),
            )

        self.schema_graph.add_nodes(nodes)
        self.schema_graph.add_cluster(nodes, key_node)

        return {node.name: node for node in nodes}

    def add_node(self, node: AtomicNode) -> AtomicNode:
        """Adds a node into the schema graph
        Raises an exception if the node is already in the graph

        Args:
            node (AtomicNode): The node to be added to the schema graph

        Returns:
            AtomicNode: The node that was added to the schema graph
        """
        self.schema_graph.check_node_not_in_graph(node)
        self.schema_graph.add_node(node)
        return node

    def add_edge(
        self, node1: SchemaNode, node2: SchemaNode, cardinality: Cardinality
    ) -> SchemaEdge:
        """Adds a directed edge between two nodes in the schema graph
        Raises an exception if either of the nodes are not in the graph

        Args:
            node1 (SchemaNode): The node at which the edge starts
            node2 (SchemaNode): The node at which the edge ends
            cardinality (Cardinality): The cardinality of the edge

        Returns:
            SchemaEdge: The edge that was added to the schema graph
        """
        edge = SchemaEdge(node1, node2, cardinality)
        self.schema_graph.add_edge(node1, node2, cardinality)
        return edge

    def map_edge_to_closure(
        self,
        edge: SchemaEdge,
        closure: Exp,
        num_args: int,
        data_relation_source: SchemaNode = None,
        data_relation_source_idxs: list[int] = None,
    ):
        """
        Maps a closure to an edge in the schema graph.
        The closure is applied when the edge is traversed.

        Args:
            edge (SchemaEdge): The edge to which the closure is to be mapped
            closure (Function): The closure to be mapped to the edge
            num_args (int): The number of arguments to the closure
            data_relation_source (SchemaNode): The source node of the data relation
            data_relation_source_idxs (list[int]): The indices of the source node in the relation

        Returns:
            None
        """

        self.backend.map_edge_to_closure(
            edge, closure, num_args, data_relation_source, data_relation_source_idxs
        )

    def create_class(self, name: str) -> SchemaClass:
        """Returns a new class that may be added to the schema graph
        A class is a set of nodes that may be treated as equivalent to each other

        Args:
            name (str): The name of the class to be created

        Returns:
            SchemaClass: The class that was instantiated
        """

        return SchemaClass(name)

    def blend(self, node1: AtomicNode, node2: AtomicNode, under: SchemaClass = None):
        """Blends two nodes in the schema graph
        If neither of the nodes has a class, then one must be provided.
        If one of the nodes has a class, then the other node is added to that class.
        If both of the nodes have classes, then they must be the same class.

        Args:
            node1 (AtomicNode): The first node to be blended
            node2 (AtomicNode): The second node to be blended
            under (SchemaClass): The class under which the nodes are to be blended

        Returns:
            None
        """

        clss1 = self.schema_graph.equivalence_class.get_classname(node1)
        clss2 = self.schema_graph.equivalence_class.get_classname(node2)

        if clss1 is None and clss2 is None and under is None:
            raise SchemaClassMustBeSpecifiedException()

        # TODO: Change this. Users should be allowed to union two classes
        if clss1 is not None and clss2 is not None and clss1 != clss2:
            raise CannotBlendNodesUnderDifferentClassesException()

        if under is not None:
            classname = under
            if node1.node_type != node2.node_type:
                raise CannotBlendNodesWithDifferentTypeException(node1, node2)
            if classname in self.schema_graph.schema_nodes and (
                clss1 is None and clss2 is None
            ):
                raise ClassAlreadyExistsException()
            if (
                classname in self.schema_graph.schema_nodes
                and (clss1 is not None and clss1 != classname)
                and (clss2 is not None and clss2 != classname)
            ):
                raise CannotBlendNodesUnderDifferentClassesException()
            under.node_type = node1.node_type
            if (
                classname not in self.schema_graph.schema_nodes
                and clss1 is None
                and clss2 is None
            ):
                domain = pd.DataFrame([], columns=[under])
                self.backend.map_atomic_node_to_domain(classname, domain)
                self.schema_graph.add_class(classname)
            new_members = frozenset()
            if clss1 is None:
                new_members = new_members.union(
                    self.schema_graph.equivalence_class.attach_classname(
                        node1, classname
                    )
                )
            if clss2 is None:
                new_members = new_members.union(
                    self.schema_graph.equivalence_class.attach_classname(
                        node2, classname
                    )
                )
            self.schema_graph.blend_nodes(node1, node2)
            self.schema_graph.blend_nodes(node1, classname)
            if new_members is not None:
                for member in new_members:
                    self.backend.extend_domain(classname, member)
        else:
            self.schema_graph.blend_nodes(node1, node2)

    # def clone(self, node: AtomicNode, name: str):
    #     equivalent_nodes = self.schema_graph.find_all_equivalent_nodes(node)
    #     already_exists = [node for node in equivalent_nodes if node.name == name]
    #     if len(already_exists) > 0:
    #         return already_exists[0]
    #     new_node = AtomicNode.clone(node, name)
    #     self.schema_graph.add_node(new_node)
    #     self.backend.clone(node, new_node)
    #     self.schema_graph.blend_nodes(new_node, node)
    #     return new_node

    def get(self, **kwargs):
        """
        Returns a table with the specified keys

        Args:
            **kwargs: A dictionary of keys and their associated nodes

        Returns:
            Table: table with the specified keys
        """
        keys = []
        with_names = []
        for k, v in kwargs.items():
            if not (isinstance(v, AtomicNode) or isinstance(v, SchemaClass)):
                raise ColumnMustBeAnAtomicNodeOrClassException(k)
            keys += [v]
            with_names += [k]

        self.schema_graph.check_nodes_in_graph(keys)
        key_set = frozenset(keys)
        diff = key_set.difference(self.schema_graph.schema_nodes)
        if len(diff) > 0:
            raise NodesDoNotExistInGraphException(list(diff))
        if with_names is None:
            with_names = [k.name for k in keys]
        key_node = SchemaNode.product(keys)
        key_nodes = SchemaNode.get_constituents(key_node)
        columns = [Domain(name, node) for name, node in zip(with_names, key_nodes)]
        return Table.construct(columns, self)

    def __find_shortest_path_in_graph(
        self, node1: SchemaNode, node2: SchemaNode, via: list[SchemaNode] = None
    ):
        return self.schema_graph.find_shortest_path(node1, node2, via)

    def does_edge_exist_in_graph(self, node1: SchemaNode, node2: SchemaNode) -> bool:
        return self.schema_graph.find_edge(node1, node2) is not None

    def find_shortest_path(
        self,
        from_columns: list[Domain],
        to_columns: list[Domain],
        via: list[SchemaNode] = None,
    ):
        """
        Finds shortest path between nodes in the schema graph,
        where the start node corresponds to the start columns
        and the end node corresponds to the end columns

        Args:
            from_columns (list[Domain]): The start columns
            to_columns (list[Domain]): The end columns
            via (list [SchemaNode]): A list of nodes to be traversed through

        Returns:
            tuple[Cardinality, list[Traversal], list[SchemaNode]]: A tuple consisting of the cardinality of the path,
            a list of traversals that need to be executed to get from the start to the end columns,
            and a list of hidden keys
        """
        node1 = SchemaNode.product([c.node for c in from_columns])
        node2 = SchemaNode.product([c.node for c in to_columns])
        cardinality, commands, hidden_keys = self.__find_shortest_path_in_graph(
            node1, node2, via
        )
        first = StartTraversal(from_columns)
        last = EndTraversal(to_columns)
        if len(commands) > 0:
            return cardinality, [first] + commands + [last], hidden_keys
        else:
            return cardinality, [first, last], hidden_keys

    def execute_query(self, table_id, derived_from, derivation):
        x, y, z, new_backend = self.backend.execute_query(
            table_id, derived_from, derivation
        )
        self.backend = new_backend
        return x, y, z, self

    def __repr__(self):
        return self.schema_graph.__repr__()

    def __str__(self):
        return self.__repr__()
