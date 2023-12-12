from collections import deque

class UnionFindAlreadyContainsItemException(Exception):
    def __init__(self, item):
        super().__init__(f"{item.val} already exists in disjoint set")


class UnionFindDoesNotContainItemException(Exception):
    def __init__(self, item):
        super().__init__(f"{item.val} does not exist in disjoint set")


class UnionFindItem:
    def __init__(self, val: any):
        self.val = val

    def __eq__(self, other):
        return self.val == other.val

    def __hash__(self):
        return hash(self.val)

    def __repr__(self):
        return str(self.val)

    def __str__(self):
        return self.__repr__()


class UnionFind:
    def __init__(self,
                 leaders: dict[UnionFindItem, UnionFindItem],
                 rank: dict[UnionFindItem, int],
                 graph: dict[UnionFindItem, frozenset[UnionFindItem]],
                 classnames: dict[UnionFindItem, UnionFindItem],
                 classname_aliases: dict[UnionFindItem, UnionFindItem]):
        self.leaders = leaders
        self.classnames = classnames
        self.classname_aliases = classname_aliases
        self.rank = rank
        self.graph = graph

    def __repr__(self):
        return str(self.leaders)

    def __contains__(self, v):
        return v in self.leaders.keys()

    @classmethod
    def initialise(cls):
        return UnionFind({}, {}, {}, {}, {})

    @classmethod
    def add_singleton(cls, uf, v, classname=None):
        item = UnionFindItem(v)
        if item in uf.leaders.keys():
            return uf
        leaders = uf.leaders | {item: item}
        rank = uf.rank | {item: 0}
        graph = uf.graph | {item: frozenset()}
        classnames = uf.classnames | {item: UnionFindItem(classname) if classname is not None else None}
        classname_aliases = uf.classname_aliases
        return UnionFind(leaders, rank, graph, classnames, classname_aliases)

    @classmethod
    def add_singletons(cls, uf, vs):
        items = frozenset([UnionFindItem(v) for v in vs])
        new_items = items.difference(uf.leaders.keys())
        leaders = uf.leaders | {item: item for item in new_items}
        rank = uf.rank | {item: 0 for item in new_items}
        graph = uf.graph | {item: frozenset() for item in new_items}
        classnames = uf.classnames | {item: None for item in new_items}
        return UnionFind(leaders, rank, graph, classnames, uf.classname_aliases)

    def find_leader(self, val):
        item = UnionFindItem(val)
        if item not in self.leaders.keys():
            raise UnionFindDoesNotContainItemException(item)
        u = item
        path = frozenset()
        while not (u == self.leaders[u]):
            path = path.union([u])
            u = self.leaders[u]
        for node in path:
            self.leaders[node] = u
            self.graph[u] = self.graph[u].union([node])
            self.graph[node] = frozenset()
        return u.val

    def attach_classname(self, val, classname):
        members = self.get_equivalence_class(val)
        for m in members:
            i = UnionFindItem(m)
            assert i not in self.classnames.keys() or self.classnames[i] is None or self.classnames[i].val == classname
            self.classnames[i] = UnionFindItem(classname) if classname is not None else None
        return members

    @classmethod
    def union(cls, uf, val1, val2):
        from schema.schema_class import SchemaClass
        item1 = UnionFindItem(val1)
        item2 = UnionFindItem(val2)
        classname_aliases = uf.classname_aliases

        if item1 not in uf.rank.keys():
            raise UnionFindDoesNotContainItemException(item1)
        if item2 not in uf.rank.keys():
            raise UnionFindDoesNotContainItemException(item2)
        rank1 = uf.rank[item1]
        clss1 = uf.classnames[item1]
        rank2 = uf.rank[item2]
        clss2 = uf.classnames[item2]
        if isinstance(val1, SchemaClass) and isinstance(val2, SchemaClass):
            if clss1 is not None:
                classname_aliases |= {item2: item1}
            else:
                classname_aliases |= {item1: item2}
            leaders = uf.leaders
            rank = uf.rank
            graph = uf.graph
        else:
            assert clss1 is None or clss2 is None or clss1 == clss2
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
        new_uf = UnionFind(leaders, rank, graph, uf.classnames, classname_aliases)
        if clss1 is not None:
            new_uf.attach_classname(val2, clss1.val)
        if clss2 is not None:
            new_uf.attach_classname(val1, clss2.val)
        return new_uf

    def get_classname(self, val):
        item = UnionFindItem(val)
        return self.classnames[item].val if item in self.classnames.keys() and self.classnames[item] is not None else None

    def get_equivalence_class(self, val) -> list[UnionFindItem]:
        from schema.schema_class import SchemaClass
        es = {val}
        if isinstance(val, SchemaClass):
            item = UnionFindItem(val)
            if item in self.classname_aliases:
                val = self.classname_aliases[item].val

            for k, v in self.classname_aliases.items():
                if v.val == val:
                    es.add(k.val)

        ldr = self.find_leader(val)
        to_explore = deque([UnionFindItem(ldr)])
        while len(to_explore) > 0:
            u = to_explore.popleft()
            es = es.union([u.val])
            ns = self.graph[u]
            for n in ns:
                to_explore.appendleft(n)
        return es
