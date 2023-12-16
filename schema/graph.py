import itertools
from collections import deque
from dataclasses import dataclass

from schema import Cardinality, SchemaEquality
from schema.edge import SchemaEdge, reverse_cardinality
from schema.edge_list import SchemaEdgeList
from schema.exceptions import AllNodesInClusterMustAlreadyBeInGraphException, \
    NodeNotInSchemaGraphException, \
    MultipleShortestPathsBetweenNodesException, CycleDetectedInPathException, \
    NoShortestPathBetweenNodesException, ClassAlreadyExistsException, NodeAlreadyInSchemaGraphException
from schema.helpers.get_indices_of_sublist import get_indices_of_sublist
from schema.helpers.is_sublist import is_sublist
from schema.node import SchemaNode
from tables.derivation import Traverse, Equate, Project, StartTraversal, EndTraversal, Cross, Expand
from union_find.union_find import UnionFind


@dataclass
class Transform:
    from_node: SchemaNode
    to_node: SchemaNode
    via: SchemaNode = None


def add_edge_to_path(edge: SchemaEdge, path: list, backwards: bool) -> list:
    if backwards:
        if edge.is_equality():
            e = SchemaEquality(edge.to_node, edge.from_node)
        else:
            e = SchemaEdge(edge.to_node, edge.from_node, reverse_cardinality(edge.cardinality))
        return [e] + path
    else:
        return path + [edge]


def is_relational(cardinality, backwards):
    return (cardinality == Cardinality.MANY_TO_MANY or
            backwards and cardinality == Cardinality.MANY_TO_ONE or
            not backwards and cardinality == Cardinality.ONE_TO_MANY)


class SchemaGraph:
    def __init__(self):
        self.adjacencyList = {}
        self.schema_nodes = []
        self.equivalence_class = UnionFind.initialise()
        self.classnames = {}

    def add_node(self, node: SchemaNode):
        if node not in frozenset(self.schema_nodes):
            self.schema_nodes += [node]
            self.equivalence_class = UnionFind.add_singleton(self.equivalence_class, node)

    def add_class(self, clss: SchemaNode):
        self.schema_nodes += [clss]
        self.equivalence_class = UnionFind.add_singleton(self.equivalence_class, clss)
        self.equivalence_class.attach_classname(clss, clss)

    def add_nodes(self, nodes: list[SchemaNode]):
        nodeset = frozenset(self.schema_nodes)
        new_nodes = list(filter(lambda n: n not in nodeset, nodes))
        self.schema_nodes += new_nodes
        self.equivalence_class = UnionFind.add_singletons(self.equivalence_class, new_nodes)

    def blend_nodes(self, node1, node2):
        self.check_nodes_in_graph([node1, node2])
        self.equivalence_class = UnionFind.union(self.equivalence_class, node1, node2)

    def check_if_class(self, name: str):
        return name[0].isupper()

    def get_node_with_name(self, name: str) -> SchemaNode:
        decomposition = name.split(".")
        if len(decomposition) == 1:
            n = SchemaNode(decomposition[0])
        else:
            cluster, name = decomposition
            n = SchemaNode(name, cluster=cluster)
        return n

    def get_node_in_graph_with_name(self, name: str) -> SchemaNode:
        n = self.get_node_with_name(name)
        if n not in self.schema_nodes:
            raise NodeNotInSchemaGraphException(n)
        else:
            return n

    def get_node_not_in_graph_with_name(self, name: str) -> SchemaNode:
        n = self.get_node_with_name(name)
        if n in self.schema_nodes:
            raise NodeAlreadyInSchemaGraphException(n)
        else:
            return n

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
            ls = list((sorted([SchemaNode.product(list(x)) for x in
                               (itertools.product(*[self.find_all_equivalent_nodes(c) for c in constituents]))],
                              key=str)))
            tr = []
            lss = set()
            for l in ls:
                if l not in lss and len(SchemaNode.get_constituents(l)) == len(constituents):
                    tr += [l]
                    lss.add(l)
            return tr

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
        neighbours = set()
        # if the node is in the adjacency list, then do a lookup
        if node in self.adjacencyList.keys():
            neighbours = SchemaEdgeList.get_edge_list(self.adjacencyList[node])
            neighbours = set([(edge.from_node, reverse_cardinality(edge.cardinality))
                              if edge.from_node != node
                              else (edge.to_node, edge.cardinality) for edge in neighbours])
        # if I can project it out
        for key in self.adjacencyList.keys():
            if is_sublist(SchemaNode.get_constituents(node), SchemaNode.get_constituents(key)):
                neighbours.add((key, Cardinality.ONE_TO_MANY))
        return neighbours

    def find_shortest_path(self, node1: SchemaNode, node2: SchemaNode, via: list[SchemaNode], backwards):
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
            nodes, edges, cmds, hks = self.find_all_shortest_paths_between_nodes(current_leg_start, current_leg_end,
                                                                                 backwards)
            if len(set(nodes).intersection(visited)) > 0:
                raise CycleDetectedInPathException()
            else:
                visited = visited.union(set(nodes))
                edge_path += edges
                commands += cmds
                hidden_keys += hks
                current_leg_start = current_leg_end
        # commands[0] = StartTraversal(node1, commands[0], explicit_keys)
        return edge_path, commands, hidden_keys

    def find_all_shortest_paths_between_nodes(self, node1: SchemaNode, node2: SchemaNode, backwards: bool = False) -> (
            bool, SchemaEdge):
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
                    shortest_path_length = c
                    shortest_paths += [path + [SchemaEquality(u, e)]] if e != u else [path]
                    nodes += [node_path + [e]] if e != u else [node_path]
                    derivation += [deriv + [Equate(u, e)]] if e != u else [deriv]
                    hidden_keys += [hks]
                # if we see a node that the goal can be projected out from,
                # then we have POTENTIALLY found a shortest path
                # adjacency list doesn't consider projections
                if node2 not in visited:
                    if not backwards and e > node2:
                        indices = get_indices_of_sublist(node2, e)
                        if u == e:
                            new_path = add_edge_to_path(SchemaEdge(e, node2, Cardinality.MANY_TO_ONE), path, backwards)
                            new_deriv = [Project(e, node2, indices)]
                            new_nodes = [node2]
                        else:
                            new_path = add_edge_to_path(SchemaEquality(u, e), path, backwards)
                            new_path = add_edge_to_path(SchemaEdge(e, node2, Cardinality.MANY_TO_ONE), new_path,
                                                        backwards)
                            new_deriv = [Equate(u, e), Project(e, node2, indices)]
                            new_nodes = [e, node2]
                        to_explore.append(
                            (node2, node_path + new_nodes, new_path, deriv + new_deriv, hks, count + len(new_nodes)))
                    if backwards and e < node2:
                        indices = get_indices_of_sublist(node2, e)
                        from schema.helpers.list_difference import list_difference
                        if u == e:
                            new_path = add_edge_to_path(SchemaEdge(e, node2, Cardinality.ONE_TO_MANY), path, backwards)
                            new_deriv = [Expand(e, node2, indices, [])]
                            new_nodes = [node2]
                        else:
                            new_path = add_edge_to_path(SchemaEquality(u, e), path, backwards)
                            new_path = add_edge_to_path(SchemaEdge(e, node2, Cardinality.ONE_TO_MANY), new_path,
                                                        backwards)
                            new_deriv = [Equate(u, e), Expand(e, node2, indices, [])]
                            new_nodes = [e, node2]
                        to_explore.append((node2, node_path + new_nodes, new_path, deriv + new_deriv, hks,
                                           count + len(new_nodes)))

            neighbours = [(e, self.get_all_neighbours_of_node(e)) for e in equivs]
            for (e, ns) in neighbours:
                for (n, c) in ns:
                    if n not in visited:
                        next_step = self.get_next_step(SchemaEdge(e, n, c), backwards)
                        if e == u:
                            new_path = add_edge_to_path(SchemaEdge(e, n, c), path, backwards)
                            to_explore.append(
                                (n, node_path + [n], new_path, deriv + [next_step],
                                 hks + next_step.hidden_keys, count + 1))
                        else:
                            new_path = add_edge_to_path(SchemaEquality(u, e), path, backwards)
                            new_path = add_edge_to_path(SchemaEdge(e, n, c), new_path, backwards)
                            to_explore.append(
                                (n, node_path + [e, n], new_path, deriv + [Equate(u, e), next_step],
                                 hks + next_step.hidden_keys, count + 2))

        if len(shortest_paths) > 1:
            raise MultipleShortestPathsBetweenNodesException(node1, node2)

        if len(shortest_paths) == 0:
            raise NoShortestPathBetweenNodesException(node1, node2)

        return nodes[0], shortest_paths[0], derivation[0], hidden_keys[0]

    def get_next_step(self, edge: SchemaEdge, backwards: bool) -> Traverse | Expand | Project:
        cardinality = edge.cardinality
        start = edge.from_node
        end = edge.to_node
        start_keys = SchemaNode.get_constituents(start)
        end_keys = SchemaNode.get_constituents(end)
        if is_relational(cardinality, backwards):
            from schema.helpers.list_difference import list_difference
            if is_sublist(start_keys, end_keys):
                hidden_keys = list_difference(start_keys, end_keys)
                indices = get_indices_of_sublist(start_keys, end_keys)
                return Expand(start, end, indices, hidden_keys)
            if is_sublist(end_keys, start_keys):
                hidden_keys = list_difference(end_keys, start_keys)
                indices = get_indices_of_sublist(end_keys, start_keys)
                return Project(start, end, indices, hidden_keys)
            else:
                return Traverse(start, end, end_keys)
        else:
            if is_sublist(start_keys, end_keys):
                indices = get_indices_of_sublist(start_keys, end_keys)
                return Expand(start, end, indices, [])
            if is_sublist(end_keys, start_keys):
                indices = get_indices_of_sublist(end_keys, start_keys)
                return Project(start, end, indices, [])
            return Traverse(start, end, [])

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
                equiv_class[i] = list(sorted(clss, key=lambda x: str(x)))
                visited = visited.union(clss)
                i += 1

        clsses = [divider + f"Class {k}" + "\n" + small_divider + "\n".join([str(x) for x in v]) + "\n" + divider for
                  k, v in
                  equiv_class.items()]
        clsses_str = "EQUIVALENCE CLASSES \n" + divider + "\n" + "\n".join(clsses) + "\n" + divider

        return adjacency_list_str + "\n" + clsses_str

    def __str__(self):
        return self.__repr__()
