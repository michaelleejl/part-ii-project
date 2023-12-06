from collections import deque

from schema import Cardinality
from schema.fully_connected import FullyConnected
from schema.edge import SchemaEdge
from schema.edge_list import SchemaEdgeList
from schema.node import SchemaNode
from union_find.union_find import UnionFind


class SchemaGraph:
    def __init__(self):
        self.adjacencyList = {}
        self.schema_nodes = []
        self.fully_connected_clusters = {}
        self.equivalence_class = UnionFind.initialise()

    def add_node(self, node: SchemaNode):
        if node not in frozenset(self.schema_nodes):
            self.schema_nodes += [node]
            self.equivalence_class = UnionFind.add_singleton(self.equivalence_class, node)

    def add_nodes(self, nodes: list[SchemaNode]):
        nodeset = frozenset(self.schema_nodes)
        new_nodes = list(filter(lambda n: n not in nodeset, nodes))
        self.schema_nodes += new_nodes
        self.equivalence_class = UnionFind.add_singletons(self.equivalence_class, new_nodes)

    def blend_nodes(self, node1, node2):
        self.equivalence_class = UnionFind.union(self.equivalence_class, node1, node2)

    def are_nodes_equal(self, node1, node2):
        return self.equivalence_class.find_leader(node1) == (self.equivalence_class.find_leader(node2))

    def add_fully_connected_cluster(self, nodes, key_node, family):
        assert frozenset(nodes) <= frozenset(self.schema_nodes)
        self.fully_connected_clusters[family] = FullyConnected(nodes, key_node)

    def add_edge(self, edge: SchemaEdge):
        from_node = edge.from_node
        to_node = edge.to_node

        if from_node == to_node:
            return
        if from_node not in self.adjacencyList:
            self.adjacencyList[from_node] = SchemaEdgeList()
        if to_node not in self.adjacencyList:
            self.adjacencyList[to_node] = SchemaEdgeList()

        self.adjacencyList[from_node] = SchemaEdgeList.add_edge(self.adjacencyList[from_node], edge)
        self.adjacencyList[to_node] = SchemaEdgeList.add_edge(self.adjacencyList[to_node], edge)

    def get_direct_edge_between_nodes(self, node1: SchemaNode, node2: SchemaNode) -> (bool, SchemaEdge):
        # Looking for direct edges.

        # case 1. node 1 = node 2 (identity)
        if node1 == node2:
            return True, SchemaEdge(node1, node2, Cardinality.ONE_TO_ONE)

        # case 2. node 1 > node 2 (projection)
        if node1 > node2:
            return True, SchemaEdge(node1, node2, Cardinality.MANY_TO_ONE)

        # case 3. node 1 < node 2 (expansion)
        if node1 < node2:
            return True, SchemaEdge(node1, node2, Cardinality.ONE_TO_MANY)

        # case 4. node 1 and node 2 equivalent
        if self.are_nodes_equal(node1, node2):
            return True, SchemaEdge(node1, node2, Cardinality.ONE_TO_ONE)

        # case 5. node 1 and node 2 are in the same cluster
        if node1.cluster is not None and node2.cluster is not None and node1.cluster == node2.cluster:
            cluster = self.fully_connected_clusters[node1]
            return True, cluster.get_edge(node1, node2)

        # case 6. edge between node 1 and node 2
        opt = list(filter(lambda e: e.to_node == node2 or e.from_node == node2, self.adjacencyList[node1]))
        if len(opt) > 0:
            return True, opt[0]
        else:
            return False, None

    def get_edge_between_nodes(self, with_transform: list[tuple[SchemaNode, SchemaNode]]) -> (bool, SchemaEdge):
        nodes = with_transform
        max_cardinality = Cardinality.ONE_TO_ONE
        edge_exists = True
        edge = None
        for (node1, node2) in nodes:
            b, c = self.get_direct_edge_between_nodes(node1, node2)
            edge_exists = edge_exists and b
            max_cardinality = max(max_cardinality, c)
        if edge_exists:
            nodes1, nodes2 = map(list, zip(*nodes))
            n1 = SchemaNode.product(nodes1)
            n2 = SchemaNode.product(nodes2)
            edge = SchemaEdge(n1, n2, max_cardinality)
        return edge_exists, edge

    def __repr__(self):
        divider = "==========================\n"
        small_divider = "--------------------------\n"
        adjacency_list = [divider + str(k) + "\n" + small_divider + str(v) + "\n" + divider for k, v in
                          self.adjacencyList.items()]
        clusters = [divider + str(k) + "\n" + small_divider + str(v) + "\n" + divider for k, v in
                    self.fully_connected_clusters.items()]

        clusters_str = "FULLY CONNECTED CLUSTERS \n" + divider + "\n" + "\n".join(clusters) + "\n" + divider
        adjacency_list_str = "ADJACENCY LIST \n" + divider + "\n" + "\n".join(adjacency_list) + "\n" + divider

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
                equiv_class[i] = clss
                visited = visited.union(clss)
                i += 1

        clsses = [divider + f"Class {k}" + "\n" + small_divider + "\n".join([str(x) for x in v]) + "\n" + divider for k, v in
                  equiv_class.items()]
        clsses_str = "EQUIVALENCE CLASSES \n" + divider + "\n" + "\n".join(clsses) + "\n" + divider

        return clusters_str + "\n" + adjacency_list_str + "\n" + clsses_str

    def __str__(self):
        return self.__repr__()
