from collections import deque
from dataclasses import dataclass

import numpy as np

from schema import Cardinality
from schema.exceptions import AllNodesInClusterMustAlreadyBeInGraphException, \
    AllNodesInFullyConnectedClusterMustHaveSameClusterException, NodeNotInSchemaGraphException, \
    FindingEdgeViaNodeMustRespectEquivalence
from schema.fully_connected import FullyConnected
from schema.edge import SchemaEdge
from schema.edge_list import SchemaEdgeList
from schema.node import SchemaNode
from union_find.union_find import UnionFind


@dataclass
class Transform:
    from_node: SchemaNode
    to_node: SchemaNode
    via: SchemaNode = None


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
        self.check_nodes_in_graph([node1, node2])
        self.equivalence_class = UnionFind.union(self.equivalence_class, node1, node2)

    def are_nodes_equal(self, node1, node2):
        self.check_nodes_in_graph([node1, node2])
        return SchemaNode.is_equivalent(node1, node2, self.equivalence_class)

    def add_fully_connected_cluster(self, nodes, key_node):
        if not (frozenset(nodes) <= frozenset(self.schema_nodes)):
            not_in_graph = frozenset(nodes).difference(frozenset(self.schema_nodes))
            raise AllNodesInClusterMustAlreadyBeInGraphException(not_in_graph)
        clusters = list(map(lambda x: x.cluster, nodes+[key_node]))
        if clusters.count(clusters[0]) != len(clusters):
            raise AllNodesInFullyConnectedClusterMustHaveSameClusterException()
        else:
            cluster = clusters[0]
        self.fully_connected_clusters[cluster] = FullyConnected(nodes, key_node)

    def check_nodes_in_graph(self, nodes: list[SchemaNode]):
        for node in nodes:
            for c in SchemaNode.get_constituents(node):
                if c and c not in self.schema_nodes:
                    raise NodeNotInSchemaGraphException(c)

    def add_edge(self, from_node: SchemaNode, to_node: SchemaNode, cardinality: Cardinality = Cardinality.MANY_TO_MANY):

        self.check_nodes_in_graph([from_node, to_node])

        if from_node == to_node:
            return
        if from_node not in self.adjacencyList:
            self.adjacencyList[from_node] = SchemaEdgeList()
        if to_node not in self.adjacencyList:
            self.adjacencyList[to_node] = SchemaEdgeList()

        edge = SchemaEdge(from_node, to_node, cardinality)

        self.adjacencyList[from_node] = SchemaEdgeList.add_edge(self.adjacencyList[from_node], edge)
        self.adjacencyList[to_node] = SchemaEdgeList.add_edge(self.adjacencyList[to_node], edge)

    def get_direct_edge_between_nodes(self, node1: SchemaNode, node2: SchemaNode, via: SchemaNode = None) -> (bool, SchemaEdge):
        self.check_nodes_in_graph([node1, node2])
        n1 = node1
        if via is not None:
            if not self.are_nodes_equal(node1, via):
                raise FindingEdgeViaNodeMustRespectEquivalence(node1, via)
            n1 = via
        # case 1. node 1 = node 2 (identity)
        if n1 == node2:
            return True, SchemaEdge(node1, node2, Cardinality.ONE_TO_ONE)

        # case 2. node 1 > node 2 (projection)
        if n1 > node2:
            return True, SchemaEdge(node1, node2, Cardinality.MANY_TO_ONE)

        # case 3. node 1 < node 2 (expansion)
        if n1 < node2:
            return True, SchemaEdge(node1, node2, Cardinality.ONE_TO_MANY)

        # case 4. node 1 and node 2 equivalent
        if SchemaNode.is_equivalent(n1, node2, self.equivalence_class):
            return True, SchemaEdge(n1, node2, Cardinality.ONE_TO_ONE)

        # case 5. node 1 and node 2 are in the same cluster
        if n1.cluster is not None and node2.cluster is not None and n1.cluster == node2.cluster and n1.cluster in self.fully_connected_clusters.keys():
            cluster = self.fully_connected_clusters[n1.cluster]
            return True, cluster.get_edge(n1, node2)

        # case 6. edge between node 1 and node 2
        if n1 in self.adjacencyList:
            opt = list(filter(lambda e: e.to_node == node2 or e.from_node == node2, self.adjacencyList[n1]))
            if len(opt) > 0:
                return True, opt[0]
        else:
            return False, None

    def get_edge_between_nodes(self, with_transform: list[Transform]) -> (bool, SchemaEdge):
        nodes = with_transform
        max_cardinality = Cardinality.MANY_TO_ONE
        edge = None
        edge_exists = True
        for transform in nodes:
            node1 = transform.from_node
            node2 = transform.to_node
            via = transform.via
            b, e = self.get_direct_edge_between_nodes(node1, node2, via)
            if not e:
                edge_exists = False
                break
            max_cardinality = Cardinality.MANY_TO_MANY if e.cardinality == Cardinality.ONE_TO_MANY or e.cardinality == Cardinality.MANY_TO_MANY else max_cardinality
        if edge_exists:
            nodes1, nodes2 = map(list, zip(*[(n.from_node, n.to_node) for n in nodes]))
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
