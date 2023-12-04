from schema.fully_connected import FullyConnected
from schema.edge import SchemaEdge
from schema.edge_list import SchemaEdgeList
from schema.node import SchemaNode
from union_find.union_find import UnionFind


class SchemaGraph:
    def __init__(self):
        self.adjacencyList = {}
        self.schema_nodes = frozenset()
        self.fully_connected_clusters = {}
        self.equivalence_class = UnionFind.initialise()

    def add_node(self, node: SchemaNode):
        if node not in self.schema_nodes:
            self.schema_nodes = self.schema_nodes.union([node])
            self.equivalence_class = UnionFind.add_singleton(self.equivalence_class, node)

    def add_nodes(self, nodes: frozenset[SchemaNode]):
        new_nodes = nodes.difference(self.schema_nodes)
        self.schema_nodes = self.schema_nodes.union(new_nodes)
        self.equivalence_class = UnionFind.add_singletons(self.equivalence_class, new_nodes)

    def blend_nodes(self, node1, node2):
        self.equivalence_class = UnionFind.union(self.equivalence_class, node1, node2)

    def are_nodes_equal(self, node1, node2):
        return self.equivalence_class.find_leader(node1).atomic_exact_equal(self.equivalence_class.find_leader(node2))

    def add_fully_connected_cluster(self, nodes, key_node, family):
        assert nodes <= self.schema_nodes
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

    def get_edge_between_nodes(self, node1: SchemaNode, node2: SchemaNode) -> (bool, SchemaEdge):
        if node1.cluster is not None and node2.cluster is not None and node1.cluster == node2.cluster:
            cluster = self.fully_connected_clusters[node1]
            return True, cluster.get_edge(node1, node2)
        opt = list(filter(lambda e: e.to_node == node2, self.adjacencyList[node1]))
        if len(opt) > 0:
            return True, opt[0]
        else:
            return False, None

    def __repr__(self):
        divider = "==========================\n"
        small_divider = "--------------------------\n"
        xs = [divider + str(k) + "\n" + small_divider + str(v) + "\n" + divider for k, v in
              self.adjacencyList.items()]
        return "\n".join(xs)

    def __str__(self):
        return self.__repr__()
