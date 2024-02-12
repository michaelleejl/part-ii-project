from __future__ import annotations

import itertools
from collections import deque

from schema.helpers.compose_cardinality import compose_cardinality
from schema import Cardinality, SchemaEquality
from schema.edge import SchemaEdge, reverse_cardinality
from schema.edge_list import SchemaEdgeList
from schema.exceptions import (
    AllNodesInClusterMustAlreadyBeInGraphException,
    NodeNotInSchemaGraphException,
    MultipleShortestPathsBetweenNodesException,
    CycleDetectedInPathException,
    NoShortestPathBetweenNodesException,
    NodeAlreadyInSchemaGraphException,
)
from schema.helpers.get_indices_of_sublist import get_indices_of_sublist
from schema.helpers.is_sublist import is_sublist
from schema.node import SchemaNode, AtomicNode, SchemaClass
from frontend.domain import Domain
from representation.representation import (
    Traverse,
    Equate,
    Project,
    Expand,
    RepresentationStep,
)
from union_find.union_find import UnionFind


def compute_cardinality_of_path(path: list[SchemaEdge]):
    c = Cardinality.ONE_TO_ONE
    for edge in path:
        c = compose_cardinality(c, edge.cardinality)
    return c


def add_edge_to_path(edge: SchemaEdge, path: list) -> list:
    return path + [edge]


def is_relational(cardinality):
    return (
        cardinality == Cardinality.MANY_TO_MANY
        or cardinality == Cardinality.ONE_TO_MANY
    )


class SchemaGraph:
    """
    A graph of schema nodes and edges. The graph is undirected and unweighted.
    """

    def __init__(self):
        """Initialises an empty schema graph."""
        self.adjacencyList: dict[SchemaNode, SchemaEdgeList] = {}
        self.schema_nodes: list[SchemaNode] = []
        self.equivalence_class: UnionFind[SchemaNode] = UnionFind.initialise()

    def add_node(self, node: SchemaNode) -> None:
        """Idempotent add of a node to the graph.

        Args:
            node (SchemaNode): The node to add to the graph

        Returns:
            None
        """
        if node not in frozenset(self.schema_nodes):
            self.schema_nodes += [node]
            self.equivalence_class = UnionFind.add_singleton(
                self.equivalence_class, node
            )

    def add_class(self, clss: SchemaClass) -> None:
        """Idempotent add of a class to the graph

        Args:
            clss (SchemaClass): The class to add to the graph

        Returns:
            None
        """
        if clss not in frozenset(self.schema_nodes):
            self.schema_nodes += [clss]
            self.equivalence_class = UnionFind.add_singleton(
                self.equivalence_class, clss
            )
            self.equivalence_class.attach_classname(clss, clss)

    def add_nodes(self, nodes: list[SchemaNode]) -> None:
        """Idempotent add of a list of nodes to the graph

        Args:
            nodes (list[SchemaNode]): The nodes to add to the graph

        Returns:
            None
        """
        nodeset = frozenset(self.schema_nodes)
        new_nodes = list(filter(lambda n: n not in nodeset, nodes))
        self.schema_nodes += new_nodes
        self.equivalence_class = UnionFind.add_singletons(
            self.equivalence_class, new_nodes
        )

    def blend_nodes(self, node1: AtomicNode, node2: AtomicNode) -> None:
        """
        Blends two atomic nodes together.

        Args:
            node1 (AtomicNode): The first atomic node to be blended
            node2 (AtomicNode): The second atomic node to be blended

        Returns:
            None
        """
        self.check_nodes_in_graph([node1, node2])
        self.equivalence_class = UnionFind.union(self.equivalence_class, node1, node2)

    def check_node_in_graph(self, n: SchemaNode) -> None:
        """
        Asserts that a node is in the graph

        Args:
            n (SchemaNode): The node to check

        Returns:
            None

        Raises:
            NodeNotInSchemaGraphException: If the node is not in the graph
        """
        ns = SchemaNode.get_constituents(n)
        for n in ns:
            if n not in self.schema_nodes:
                raise NodeNotInSchemaGraphException(n)

    def check_node_not_in_graph(self, n: SchemaNode) -> None:
        """
        Asserts that a node is not in the graph

        Args:
            n (SchemaNode): The node to check

        Returns:
            None

        Raises:
            NodeAlreadyInSchemaGraphException: If the node is in the graph
        """
        ns = SchemaNode.get_constituents(n)
        for n in ns:
            if n in self.schema_nodes:
                raise NodeAlreadyInSchemaGraphException(n)

    def are_nodes_equal(self, node1: SchemaNode, node2: SchemaNode) -> bool:
        """
        Returns True if the two nodes are equivalent, False otherwise

        Args:
            node1 (SchemaNode): The first node
            node2 (SchemaNode): The second node

        Returns:
            bool: True if the two nodes are equivalent, False otherwise
        """
        self.check_nodes_in_graph([node1, node2])
        return SchemaNode.is_equivalent(node1, node2, self.equivalence_class)

    def add_cluster(self, nodes: list[SchemaNode], key_node: SchemaNode) -> None:
        """
        Adds a star-shaped cluster of nodes to the graph. The key node is the central node in the cluster.

        Args:
            nodes (list[SchemaNode]): The nodes in the cluster
            key_node (SchemaNode): The key node in the cluster

        Returns:
            None
        """
        if not (frozenset(nodes) <= frozenset(self.schema_nodes)):
            not_in_graph = frozenset(nodes).difference(frozenset(self.schema_nodes))
            raise AllNodesInClusterMustAlreadyBeInGraphException(not_in_graph)
        for node in nodes:
            self.add_edge(key_node, node, Cardinality.MANY_TO_ONE)

    def find_all_equivalent_nodes(self, node: SchemaNode) -> list[SchemaNode]:
        """
        Returns all nodes in the graph that are equivalent to the given node

        Args:
            node (SchemaNode): The node to find equivalent nodes to

        Returns:
            set[SchemaNode]: A list of all nodes in the graph that are equivalent to the given node
        """
        constituents = SchemaNode.get_constituents(node)
        # if node atomic
        if len(constituents) == 1:
            return list(self.equivalence_class.get_equivalence_class(node))
        else:
            ls = list(
                (
                    sorted(
                        [
                            SchemaNode.product(list(x))
                            for x in (
                                itertools.product(
                                    *[
                                        self.find_all_equivalent_nodes(c)
                                        for c in constituents
                                    ]
                                )
                            )
                        ],
                        key=str,
                    )
                )
            )
            tr = []
            lss = set()
            for l in ls:
                if l not in lss and len(SchemaNode.get_constituents(l)) == len(
                    constituents
                ):
                    tr += [l]
                    lss.add(l)
            return tr

    def check_nodes_in_graph(self, nodes: list[SchemaNode]) -> None:
        """
        Asserts that all nodes are in the graph

        Args:
            nodes (list[SchemaNode]): The nodes to check

        Returns:
            None
        """
        for node in nodes:
            self.check_node_in_graph(node)

    def add_edge(
        self,
        from_node: SchemaNode,
        to_node: SchemaNode,
        cardinality: Cardinality = Cardinality.MANY_TO_MANY,
    ) -> None:
        """
        Adds an edge to the graph. If the nodes are not in the graph, an exception is raised.

        Args:
            from_node (SchemaNode): The node the edge is from
            to_node (SchemaNode): The node the edge is to
            cardinality (Cardinality): The cardinality of the edge

        Returns:
            None

        Raises:
            NodeNotInSchemaGraphException: If the from_node or to_node are not in the graph
        """
        self.check_nodes_in_graph([from_node, to_node])

        if from_node == to_node:
            return
        if from_node not in self.adjacencyList:
            self.adjacencyList[from_node] = SchemaEdgeList()
        if to_node not in self.adjacencyList:
            self.adjacencyList[to_node] = SchemaEdgeList()

        edge = SchemaEdge(from_node, to_node, cardinality)

        self.adjacencyList[from_node] = SchemaEdgeList.add_edge(
            self.adjacencyList[from_node], edge
        )
        self.adjacencyList[to_node] = SchemaEdgeList.add_edge(
            self.adjacencyList[to_node], SchemaEdge.invert(edge)
        )

    def find_edge(self, from_node: SchemaNode, to_node: SchemaNode) -> SchemaEdge | None:
        """
        Finds an edge between two nodes in the graph

        Args:
            from_node (SchemaNode): The node the edge is from
            to_node (SchemaNode): The node the edge is to

        Returns:
            SchemaEdge | None: The edge between the two nodes, or None if no such edge exists
        """
        self.check_nodes_in_graph([from_node, to_node])
        if from_node in self.adjacencyList:
            edges = SchemaEdgeList.get_edge_list(self.adjacencyList[from_node])
            for edge in edges:
                if edge.to_node == to_node:
                    return edge
        return None

    def get_all_neighbours_of_node(
        self, node: SchemaNode
    ) -> set[tuple[SchemaNode, Cardinality]]:
        """
        Returns all neighbours of a node in the graph, as well as the cardinality of the edge between the node and its neighbour

        Args:
            node (SchemaNode): The node to find the neighbours of

        Returns:
            set[tuple[SchemaNode, Cardinality]]: A set of tuples, where each tuple is of the form (neighbour, cardinality)
        """
        neighbours = set()
        # if the node is in the adjacency list, then do a lookup
        if node in self.adjacencyList.keys():
            neighbours = SchemaEdgeList.get_edge_list(self.adjacencyList[node])
            neighbours = set(
                [
                    (
                        (edge.from_node, reverse_cardinality(edge.cardinality))
                        if edge.from_node != node
                        else (edge.to_node, edge.cardinality)
                    )
                    for edge in neighbours
                ]
            )
        # if I can do an expansion / cross product
        for key in self.adjacencyList.keys():
            if is_sublist(
                SchemaNode.get_constituents(node), SchemaNode.get_constituents(key)
            ):
                neighbours.add((key, Cardinality.ONE_TO_MANY))
        return neighbours

    def find_shortest_path(
        self, node1: SchemaNode, node2: SchemaNode, via: list[SchemaNode] | None
    ) -> tuple[Cardinality, list[RepresentationStep], list[Domain]]:
        """
        Finds the shortest path between two nodes in the graph.
        If waypoints are specified, the path will find the shortest path between the two nodes that
        passes through the waypoints.

        Args:
            node1 (SchemaNode): The first node
            node2 (SchemaNode): The second node
            via (list[SchemaNode] | None): The waypoints

        Returns:
            tuple[Cardinality, list[RepresentationStep], list[Domain]]: A tuple of the form (cardinality,
            commands, hidden_keys), where cardinality is the cardinality of the path, commands is a list of
            commands for generating the path, and hidden_keys is a list of hidden keys in the path

        Raises:
            NodeNotInSchemaGraphException: If the node1, node2, or any of the waypoints are not in the graph
            NoShortestPathBetweenNodesException: If there is no shortest path between the nodes
            MultipleShortestPathsBetweenNodesException: If there are multiple shortest paths between the nodes
        """
        if via is None:
            waypoints = []
        else:
            waypoints = via
        self.check_nodes_in_graph([node1, node2] + waypoints)
        current_leg_start = node1
        visited = {node1}
        edge_path, commands, hidden_keys = [], [], []
        for i in range(0, len(waypoints) + 1):
            if i >= len(waypoints):
                current_leg_end = node2
            else:
                current_leg_end = waypoints[i]
            nodes, edges, cmds, hks = self.find_all_shortest_paths_between_nodes(
                current_leg_start, current_leg_end
            )
            if len(set(nodes).intersection(visited)) > 0:
                raise CycleDetectedInPathException()
            else:
                visited = visited.union(set(nodes))
                edge_path += edges
                commands += cmds
                hidden_keys += hks
                current_leg_start = current_leg_end
        return compute_cardinality_of_path(edge_path), commands, hidden_keys

    def find_all_shortest_paths_between_nodes(
        self, node1: SchemaNode, node2: SchemaNode
    ) -> (list[SchemaNode], list[SchemaEdge], list[RepresentationStep], list[Domain]):
        """
        Finds the shortest path between two nodes in the graph

        Args:
            node1 (SchemaNode): The first node
            node2 (SchemaNode): The second node

        Returns:
            list[SchemaNode]: A list of nodes in the shortest path
            list[SchemaEdge]: A list of edges in the shortest path
            list[RepresentationStep]: A list of commands for generating the shortest path
            list[Domain]: A list of hidden keys in the shortest path

        Raises:
            NoShortestPathBetweenNodesException: If there is no shortest path between the nodes
            MultipleShortestPathsBetweenNodesException: If there are multiple shortest paths between the nodes
        """

        to_explore = deque()
        visited = {node1}
        to_explore.append((node1, [], [], [], [], 0))

        shortest_paths = []
        shortest_path_length = -1

        derivation = []
        nodes = []
        hidden_keys = []

        while len(to_explore) > 0:
            u, node_path, path, deriv, hks, count = to_explore.popleft()
            # by the BFS invariant, if we are considering
            # nodes with a path length > than the shortest path length
            # we will never find another shortest path
            if 0 < shortest_path_length < count:
                break

            equivs = self.find_all_equivalent_nodes(u)
            visited = visited.union(equivs)
            # if we see the goal, then we have found a shortest path
            for e in equivs:
                if e == node2:
                    if e != u:
                        c = count + 1
                    else:
                        c = count
                    if 0 < shortest_path_length < c:
                        continue
                    if c < shortest_path_length:
                        shortest_paths = []
                    shortest_path_length = c
                    shortest_paths += (
                        [path + [SchemaEquality(u, e)]] if e != u else [path]
                    )
                    nodes += [node_path + [e]] if e != u else [node_path]
                    derivation += [deriv + [Equate(u, e)]] if e != u else [deriv]
                    hidden_keys += [hks]
                # if we see a node that the goal can be projected out from,
                # then we have POTENTIALLY found a shortest path
                # adjacency list doesn't consider projections
                if node2 not in visited:
                    # Consider not inferring projections
                    if e > node2:
                        c1 = SchemaNode.get_constituents(e)
                        c2 = SchemaNode.get_constituents(node2)
                        indices = get_indices_of_sublist(c2, c1)
                        if u == e:
                            new_path = add_edge_to_path(
                                SchemaEdge(e, node2, Cardinality.MANY_TO_ONE), path
                            )
                            new_deriv = [Project(e, node2, indices)]
                            new_nodes = [node2]
                        else:
                            new_path = add_edge_to_path(SchemaEquality(u, e), path)
                            new_path = add_edge_to_path(
                                SchemaEdge(e, node2, Cardinality.MANY_TO_ONE), new_path
                            )
                            new_deriv = [Equate(u, e), Project(e, node2, indices)]
                            new_nodes = [e, node2]
                        to_explore.append(
                            (
                                node2,
                                node_path + new_nodes,
                                new_path,
                                deriv + new_deriv,
                                hks,
                                count + len(new_nodes),
                            )
                        )

            neighbours = [(e, self.get_all_neighbours_of_node(e)) for e in equivs]
            for e, ns in neighbours:
                for n, c in ns:
                    if n not in visited:
                        next_step = self.get_next_step(SchemaEdge(e, n, c))
                        if e == u:
                            new_path = add_edge_to_path(SchemaEdge(e, n, c), path)
                            to_explore.append(
                                (
                                    n,
                                    node_path + [n],
                                    new_path,
                                    deriv + [next_step],
                                    hks + next_step.get_hidden_keys(),
                                    count + 1,
                                )
                            )
                        else:
                            new_path = add_edge_to_path(SchemaEquality(u, e), path)
                            new_path = add_edge_to_path(SchemaEdge(e, n, c), new_path)
                            to_explore.append(
                                (
                                    n,
                                    node_path + [e, n],
                                    new_path,
                                    deriv + [Equate(u, e), next_step],
                                    hks + next_step.get_hidden_keys(),
                                    count + 2,
                                )
                            )

        if len(shortest_paths) > 1:
            raise MultipleShortestPathsBetweenNodesException(
                node1, node2, shortest_paths
            )

        if len(shortest_paths) == 0:
            raise NoShortestPathBetweenNodesException(node1, node2)

        return nodes[0], shortest_paths[0], derivation[0], hidden_keys[0]

    def get_next_step(self, edge: SchemaEdge) -> Traverse | Expand | Project:
        """
        Returns the next command in the list of commands, given the edge

        Args:
            edge (SchemaEdge): The edge

        Returns:
            Traverse | Expand | Project: The next command in the list of commands
        """
        cardinality = edge.cardinality
        start = edge.from_node
        end = edge.to_node
        start_keys = SchemaNode.get_constituents(start)
        end_keys = SchemaNode.get_constituents(end)
        if not edge.is_functional():
            from schema.helpers.list_difference import list_difference

            if is_sublist(start_keys, end_keys):
                hidden_keys = list_difference(end_keys, start_keys)
                indices = get_indices_of_sublist(start_keys, end_keys)
                return Expand(start, end, indices, [Domain(node.name, node) for node in hidden_keys])
            else:
                return Traverse(edge)
        else:
            if is_sublist(start_keys, end_keys):
                indices = get_indices_of_sublist(start_keys, end_keys)
                return Expand(start, end, indices, [])
            if is_sublist(end_keys, start_keys):
                indices = get_indices_of_sublist(end_keys, start_keys)
                return Project(start, end, indices)
            return Traverse(edge)

    def __repr__(self):
        divider = "==========================\n"
        small_divider = "--------------------------\n"
        adjacency_list = [
            divider + str(k) + "\n" + small_divider + str(v) + "\n" + divider
            for k, v in self.adjacencyList.items()
        ]

        adjacency_list_str = (
            "ADJACENCY LIST \n"
            + divider
            + "\n"
            + "\n".join(adjacency_list)
            + "\n"
            + divider
        )

        ns = deque(self.schema_nodes)
        visited = frozenset()
        i = 0
        equiv_class = {}
        while len(ns) > 0:
            u = ns.popleft()
            while u in visited:
                if len(ns) == 0:
                    break
                u = ns.popleft()
            if u not in visited:
                clss = self.equivalence_class.get_equivalence_class(u)
                equiv_class[i] = list(sorted(clss, key=lambda x: str(x)))
                visited = visited.union(clss)
                i += 1

        clsses = [
            divider
            + f"Class {k}"
            + "\n"
            + small_divider
            + "\n".join([str(x) for x in v])
            + "\n"
            + divider
            for k, v in equiv_class.items()
        ]
        clsses_str = (
            "EQUIVALENCE CLASSES \n"
            + divider
            + "\n"
            + "\n".join(clsses)
            + "\n"
            + divider
        )

        return adjacency_list_str + "\n" + clsses_str

    def __str__(self):
        return self.__repr__()
