class SchemaNodeNameShouldNotContainSemicolonException(Exception):
    def __init__(self, name):
        super().__init__(f"Schema node name should not contain a semicolon. Name: f{name}")


class SchemaNode:
    def __init__(self, name: str, constituents: frozenset[any] = None, cluster: str = None, graph = None):
        if ";" in name and constituents is None:
            raise SchemaNodeNameShouldNotContainSemicolonException(name)
        self.name = name
        self.constituents = constituents
        self.cluster = cluster
        self.key = (self.name, self.cluster)
        self.graph = graph

    def prepend_id(self, val: str) -> str:
        return f"{hash(self.key)}_{val}"

    def get_key(self):
        return self.key

    @classmethod
    def product(cls, nodes: frozenset[any]):
        atomics = frozenset()
        for node in nodes:
            atomics = atomics.union(SchemaNode.get_constituents(node))
        if len(atomics) == 1:
            x, = atomics
            return x
        name = "; ".join([a.name for a in atomics])
        clusters = frozenset([n.cluster for n in nodes])
        cluster = None
        if len(clusters) == 1:
            cluster, = clusters
        constituents = atomics
        return SchemaNode(name, constituents, cluster)

    @classmethod
    def get_constituents(cls, node):
        c = node.constituents
        if c is None:
            return frozenset([node])
        else:
            return c

    def __hash__(self):
        if self.constituents is None:
            return hash(self.get_key())
        else:
            return hash(self.constituents)

    def atomic_exact_equal(self, other):
        return self.constituents is None and other.constituents is None and (self.get_key() == other.get_key())

    def __eq__(self, other):
        if isinstance(other, SchemaNode):
            # check if it's atomic
            if self.constituents is None:
                return other.constituents is None and self.get_key() == other.get_key()
            else:
                return self.constituents == other.constituents
        return NotImplemented

    def __le__(self, other):
        if isinstance(other, SchemaNode):
            return SchemaNode.get_constituents(self) <= SchemaNode.get_constituents(other)
        return NotImplemented

    def __lt__(self, other):
        if isinstance(other, SchemaNode):
            return SchemaNode.get_constituents(self) < SchemaNode.get_constituents(other)
        return NotImplemented

    def __ge__(self, other):
        if isinstance(other, SchemaNode):
            return SchemaNode.get_constituents(self) >= SchemaNode.get_constituents(other)
        return NotImplemented

    def __gt__(self, other):
        if isinstance(other, SchemaNode):
            return SchemaNode.get_constituents(self) > SchemaNode.get_constituents(other)
        return NotImplemented

    def __repr__(self):
        return f"{self.cluster}.{self.name}" if self.cluster is not None else self.name

    def __str__(self):
        return self.__repr__()
