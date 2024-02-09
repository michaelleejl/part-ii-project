from __future__ import annotations
from collections import deque
from typing import TypeVar, Generic
T = TypeVar('T')


class UnionFindAlreadyContainsItemException(Exception):
    def __init__(self, item):
        super().__init__(f"{item.val} already exists in disjoint set")


class UnionFindDoesNotContainItemException(Exception):
    def __init__(self, item):
        super().__init__(f"{item.val} does not exist in disjoint set")


class UnionFindItem(Generic[T]):
    """
    A class to represent an item in a UnionFind data structure
    Parameterised on T
    """

    def __init__(self, val: T):
        """
        Creates a new UnionFindItem with the given value

        Args:
            val: The value of the item
        """
        self.val = val

    def __eq__(self, other: UnionFindItem[T]):
        return self.val == other.val

    def __hash__(self):
        return hash(self.val)

    def __repr__(self):
        return str(self.val)

    def __str__(self):
        return self.__repr__()


class UnionFind(Generic[T]):
    """
    A UnionFind data structure
    Parameterised on type T
    """
    def __init__(self,
                 leaders: dict[UnionFindItem[T], UnionFindItem[T]],
                 rank: dict[UnionFindItem[T], int],
                 graph: dict[UnionFindItem[T], frozenset[UnionFindItem[T]]],
                 classnames: dict[UnionFindItem[T], UnionFindItem[T]]):
        self.leaders: dict[UnionFindItem[T], UnionFindItem[T]] = leaders
        self.classnames: dict[UnionFindItem[T], UnionFindItem[T]] = classnames
        self.rank: dict[UnionFindItem[T], int] = rank
        self.graph: dict[UnionFindItem[T], frozenset[UnionFindItem[T]]] = graph

    def __repr__(self):
        return str(self.leaders)

    def __contains__(self, v):
        return v in self.leaders.keys()

    @classmethod
    def initialise(cls):
        return UnionFind({}, {}, {}, {})

    @classmethod
    def add_singleton(cls, uf: UnionFind[T], v:T, classname: str | None = None) -> UnionFind[T]:
        """
        Adds a singleton to the UnionFind data structure

        Args:
            uf (UnionFind[T]): The UnionFind data structure
            v (T): The value of the singleton to be addled
            classname (str | None): The class name of the singleton

        Returns:
            UnionFind[T]: A new UnionFind data structure with the added singleton
        """
        item = UnionFindItem(v)
        if item in uf.leaders.keys():
            return uf
        leaders = uf.leaders | {item: item}
        rank = uf.rank | {item: 0}
        graph = uf.graph | {item: frozenset()}
        classnames = uf.classnames | {item: UnionFindItem(classname) if classname is not None else None}
        return UnionFind(leaders, rank, graph, classnames)

    @classmethod
    def add_singletons(cls, uf: UnionFind[T], vs: list[T]) -> UnionFind[T]:
        """
        Adds union find items to the UnionFind data structure

        Args:
            uf (UnionFind[T]): The UnionFind data structure
            vs (T): The values of the singletons to be addled

        Returns:
            UnionFind[T]: A new UnionFind data structure with the added singleton
        """
        items = frozenset([UnionFindItem(v) for v in vs])
        new_items = items.difference(uf.leaders.keys())
        leaders = uf.leaders | {item: item for item in new_items}
        rank = uf.rank | {item: 0 for item in new_items}
        graph = uf.graph | {item: frozenset() for item in new_items}
        classnames = uf.classnames | {item: None for item in new_items}
        return UnionFind(leaders, rank, graph, classnames)

    def find_leader(self, val: T) -> T:
        """
        Finds the leader of the equivalence class of the given value

        Args:
            val (T): The value

        Returns:
            T: The leader of the equivalence class of the given value
        """
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

    def attach_classname(self, val: T, classname: str) -> list[T]:
        """
        Attaches a class name to the equivalence class of val

        Args:
            val (T): The value in the equivalence class
            classname (str): The class name to be attached to the class

        Returns:
            list[T]: The members of the equivalence class
        """
        members = self.get_equivalence_class(val)
        for m in members:
            i = UnionFindItem(m)
            assert i not in self.classnames.keys() or self.classnames[i] is None or self.classnames[i].val == classname
            self.classnames[i] = UnionFindItem(classname) if classname is not None else None
        return members

    @classmethod
    def union(cls, uf: UnionFind[T], val1: T, val2: T) -> UnionFind[T]:
        """
        Unions the equivalence classes of val1 and val2

        Args:
            uf (UnionFind[T]): The UnionFind data structure
            val1 (T): The first value
            val2 (T): The second value

        Returns:
            UnionFind[T]: A new UnionFind data structure with the equivalence classes of val1 and val2 unioned
        """
        """
        :param uf: 
        :param val1: 
        :param val2: 
        :return: 
        """
        item1 = UnionFindItem(val1)
        item2 = UnionFindItem(val2)

        if item1 not in uf.rank.keys():
            raise UnionFindDoesNotContainItemException(item1)
        if item2 not in uf.rank.keys():
            raise UnionFindDoesNotContainItemException(item2)
        rank1 = uf.rank[item1]
        clss1 = uf.classnames[item1]
        rank2 = uf.rank[item2]
        clss2 = uf.classnames[item2]
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
        new_uf = UnionFind(leaders, rank, graph, uf.classnames)
        if clss1 is not None:
            new_uf.attach_classname(val2, clss1.val)
        if clss2 is not None:
            new_uf.attach_classname(val1, clss2.val)
        return new_uf

    def get_classname(self, val: T) -> str | None:
        """
        Gets the class name of the equivalence class of the given value, or None if there is no such class name

        Args:
            val (T): The value

        Returns:
            str | None: The class name of the equivalence class of the given value, or None if there is no such class name
        """

        item = UnionFindItem(val)
        return self.classnames[item].val if item in self.classnames.keys() and self.classnames[item] is not None else None

    def get_equivalence_class(self, val: T) -> set[T]:
        """
        Gets the equivalence class of the given value

        Args:
            val (T): The value

        Returns:
            set[T]: The equivalence class of the given value
        """
        es = {val}
        ldr = self.find_leader(val)
        to_explore = deque([UnionFindItem(ldr)])
        while len(to_explore) > 0:
            u = to_explore.popleft()
            es = es.union([u.val])
            ns = self.graph[u]
            for n in ns:
                to_explore.appendleft(n)
        return es
