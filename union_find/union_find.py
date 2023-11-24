from collections import deque

from schema import SchemaNode


class UnionFindAlreadyContainsItemException(Exception):
    def __init__(self, item):
        super().__init__(f"{item.val} already exists in disjoint set")


class UnionFindDoesNotContainItemException(Exception):
    def __init__(self, item):
        super().__init__(f"{item.val} does not exist in disjoint set")


class UnionFindItem:
    def __init__(self, val: any, node: SchemaNode):
        self.val = val
        self.node = node

    def __eq__(self, other):
        return self.val == other.val and self.node == other.node

    def __hash__(self):
        return hash((self.val, self.node))

    def __repr__(self):
        return str(self.val) + f"({self.node})"

    def __str__(self):
        return self.__repr__()


class UnionFind:
    def __init__(self, leaders: dict[UnionFindItem, UnionFindItem], rank: dict[UnionFindItem, int], graph: dict[UnionFindItem, frozenset[UnionFindItem]]):
        self.leaders = leaders
        self.rank = rank
        self.graph = graph

    def __repr__(self):
        return str(self.leaders)

    def __contains__(self, v):
        return v in self.leaders.keys()

    @classmethod
    def initialise(cls):
        return UnionFind({}, {}, {})

    @classmethod
    def add_singleton(cls, uf, v, node):
        item = UnionFindItem(v, node)
        if item in uf.leaders.keys():
            return uf
        leaders = uf.leaders | {item: item}
        rank = uf.rank | {item: 0}
        graph = uf.graph | {item: frozenset()}
        return UnionFind(leaders, rank, graph)

    @classmethod
    def add_singletons(cls, uf, vs, node):
        items = frozenset([UnionFindItem(v, node) for v in vs])
        new_items = items.difference(uf.leaders.keys())
        leaders = uf.leaders | {item: item for item in new_items}
        rank = uf.rank | {item: 0 for item in new_items}
        graph = uf.graph | {item: frozenset() for item in new_items}
        return UnionFind(leaders, rank, graph)

    def find_leader(self, item: UnionFindItem) -> UnionFindItem:
        if item not in self.leaders.keys():
            raise UnionFindDoesNotContainItemException(item)
        u = item
        path = frozenset()
        while u != self.leaders[u]:
            path = path.union([u])
            u = self.leaders[u]
        for node in path:
            self.leaders[node] = u
            self.graph[u] = self.graph[u].union([node])
            self.graph[node] = frozenset()
        return u

    @classmethod
    def union(cls, uf, item1: UnionFindItem, item2: UnionFindItem):
        if item1 not in uf.rank.keys():
            raise UnionFindDoesNotContainItemException(item1)
        if item2 not in uf.rank.keys():
            raise UnionFindDoesNotContainItemException(item2)
        rank1 = uf.rank[item1]
        rank2 = uf.rank[item2]
        leaders = uf.leaders
        graph = uf.graph
        rank = uf.rank
        if rank1 > rank2:
            leaders[item2] = item1
            graph[item1] = graph[item1].union([item2])
        elif rank2 > rank1:
            leaders[item1] = item2
            graph[item2] = graph[item2].union([item1])
        else:
            leaders[item2] = item1
            graph[item1] = graph[item1].union([item2])
            rank[item1] += 1
        return UnionFind(leaders, rank, graph)

    def get_equivalence_class(self, item: UnionFindItem) -> frozenset[UnionFindItem]:
        ldr = self.find_leader(item)
        es = set()
        to_explore = deque([ldr])
        while len(to_explore) > 0:
            u = to_explore.popleft()
            es = es.union([u])
            ns = self.graph[u]
            for n in ns:
                to_explore.appendleft(n)
        return frozenset(es)
