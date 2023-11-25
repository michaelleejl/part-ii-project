from schema.edge import SchemaEdge
from schema.edge_list import SchemaEdgeList
from schema.node import SchemaNode


class SchemaGraph:
    def __init__(self):
        self.adjacencyList = {}
        self.schema_nodes = frozenset()

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

    def add_node(self, node: SchemaNode):
        if node not in self.schema_nodes:
            self.schema_nodes = self.schema_nodes.union([node])

    def add_nodes(self, nodes: frozenset[SchemaNode]):
        if len(nodes.difference(self.schema_nodes)) == len(nodes):
            self.schema_nodes = self.schema_nodes.union(nodes)

    def replace_node(self, node: SchemaNode):
        if node in self.schema_nodes:
            self.schema_nodes = self.schema_nodes.difference([node]).union([node])

    def get_node(self, name: str, family: str) -> SchemaNode:
        ns = list(filter(lambda x: x.get_key() == (name, family), self.schema_nodes))
        if len(ns) > 0:
            return ns[0]

    def does_relation_exist(self, node1: SchemaNode, node2: SchemaNode):
        return node1 in self.adjacencyList and len(list(filter(lambda e: e.to_node == node2, self.adjacencyList[node1]))) > 0

    def get_edge_between_nodes(self, node1: SchemaNode, node2: SchemaNode):
        return list(filter(lambda e: e.to_node == node2, self.adjacencyList[node1]))[0]

    def __repr__(self):
        divider = "==========================\n"
        small_divider = "--------------------------\n"
        xs = [divider + str(k) + "\n" + small_divider + str(v) + "\n" + divider for k, v in
              self.adjacencyList.items()]
        return "\n".join(xs)

    def __str__(self):
        return self.__repr__()
