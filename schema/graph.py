import itertools
from collections import deque
from dataclasses import dataclass

from schema import Cardinality, SchemaEquality
from schema.schema_class import SchemaClass
from schema.edge import SchemaEdge, reverse_cardinality
from schema.edge_list import SchemaEdgeList
from schema.exceptions import AllNodesInClusterMustAlreadyBeInGraphException, \
    NodeNotInSchemaGraphException, \
    MultipleShortestPathsBetweenNodesException, CycleDetectedInPathException, \
    NoShortestPathBetweenNodesException, ClassAlreadyExistsException
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

    def blend_nodes(self, node1, node2, under: str = None):
        self.check_nodes_in_graph([node1, node2])
        self.equivalence_class = UnionFind.union(self.equivalence_class, node1, node2)
        classname = SchemaClass(under)
        if classname in self.schema_nodes:
            raise ClassAlreadyExistsException()
        else:
            self.add_node(classname)
            self.equivalence_class = UnionFind.union(self.equivalence_class, node1, classname)

    def are_nodes_equal(self, node1, node2):
        self.check_nodes_in_graph([node1, node2])
        return SchemaNode.is_equivalent(node1, node2, self.equivalence_class)

    def add_cluster(self, nodes, key_node):
        if not (frozenset(nodes) <= frozenset(self.schema_nodes)):
            not_in_graph = frozenset(nodes).difference(frozenset(self.schema_nodes))
            raise AllNodesInClusterMustAlreadyBeInGraphException(not_in_graph)
        for node in nodes:
            self.add_edge(key_node, node, Cardinality.MANY_TO_ONE)

    def find_all_equivalent_nodes(self, node):
        constituents = SchemaNode.get_constituents(node)
        # if node atomic
        if len(constituents) == 1:
            return list(self.equivalence_class.get_equivalence_class(node))
        else:
            return list(set([SchemaNode.product(list(x)) for x in
                    (itertools.product(*[self.find_all_equivalent_nodes(c) for c in constituents]))]))

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

    def get_all_neighbours_of_node(self, node):
        if node in self.adjacencyList.keys():
            neighbours = SchemaEdgeList.get_edge_list(self.adjacencyList[node])
            return [(edge.from_node, reverse_cardinality(edge.cardinality))
                    if edge.from_node != node
                    else (edge.to_node, edge.cardinality) for edge in neighbours]
        else:
            return []

    def find_shortest_path(self, node1: SchemaNode, node2: SchemaNode, via: list[SchemaNode] = None):
        if via is None:
            waypoints = []
        else:
            waypoints = via
        self.check_nodes_in_graph([node1, node2] + waypoints)
        current_leg_start = node1
        visited = {node1}
        node_path, edge_path = [], []
        for i in range(0, len(waypoints) + 1):
            if i >= len(waypoints):
                current_leg_end = node2
            else:
                current_leg_end = waypoints[i]
            nodes, edges = self.find_all_shortest_paths_between_nodes(current_leg_start, current_leg_end)
            print(set(nodes).intersection(visited))
            if len(set(nodes).intersection(visited)) > 0:
                raise CycleDetectedInPathException()
            else:
                visited = visited.union(set(nodes))
                node_path += nodes
                edge_path += edges
                current_leg_start = current_leg_end
        return node_path, edge_path

    def find_all_shortest_paths_between_nodes(self, node1: SchemaNode, node2: SchemaNode) -> (bool, SchemaEdge):
        to_explore = deque()
        visited = {node1}
        to_explore.append((node1, [], [], 0))

        shortest_paths = []
        shortest_path_length = -1

        edge_traversal = []

        while len(to_explore) > 0:
            u, path, edges, count = to_explore.popleft()
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
                    shortest_path_length = count
                    shortest_paths += [path + [e]] if e != u else [path]
                    edge_traversal += [edges + [SchemaEquality(u, e)]] if e != u else [edges]
                # if we see a node that the goal can be projected out from,
                # then we have POTENTIALLY found a shortest path
                # adjacency list doesn't consider projections
                if e > node2 and node2 not in visited:
                    if u == e:
                        to_explore.append((node2, path + [node2], edges + [SchemaEdge(e, node2, Cardinality.MANY_TO_ONE)],
                                           count + 1))
                    else:
                        to_explore.append((node2, path + [e, node2], edges + [SchemaEquality(u, e), SchemaEdge(e, node2, Cardinality.MANY_TO_ONE)], count + 1))

            neighbours = [(e, self.get_all_neighbours_of_node(e)) for e in equivs]
            for (e, ns) in neighbours:
                for (n, c) in ns:
                    if n not in visited:
                        if e == u:
                            to_explore.append((n, path + [n], edges + [SchemaEdge(e, n, c)], count + 1))
                        else:
                            to_explore.append((n, path + [e, n], edges + [SchemaEquality(u, e), SchemaEdge(e, n, c)], count + 1))

        if len(shortest_paths) > 1:
            raise MultipleShortestPathsBetweenNodesException(node1, node2)

        if len(shortest_paths) == 0:
            raise NoShortestPathBetweenNodesException(node1, node2)

        return shortest_paths[0], edge_traversal[0]

    def __repr__(self):
        divider = "==========================\n"
        small_divider = "--------------------------\n"
        adjacency_list = [divider + str(k) + "\n" + small_divider + str(v) + "\n" + divider for k, v in
                          self.adjacencyList.items()]

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

        clsses = [divider + f"Class {k}" + "\n" + small_divider + "\n".join([str(x) for x in v]) + "\n" + divider for
                  k, v in
                  equiv_class.items()]
        clsses_str = "EQUIVALENCE CLASSES \n" + divider + "\n" + "\n".join(clsses) + "\n" + divider

        return adjacency_list_str + "\n" + clsses_str

    def __str__(self):
        return self.__repr__()
