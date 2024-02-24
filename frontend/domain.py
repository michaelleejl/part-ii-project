from schema.node import SchemaNode, AtomicNode, SchemaClass


class Domain:
    """
    A domain is a named alias for a node in the schema
    """

    def __init__(self, name: str, node: AtomicNode | SchemaClass):
        """
        Initialises a domain
        Args:
             name(str): The name of the domain
             node(SchemaNode): The node in the schema
        """
        assert len(SchemaNode.get_constituents(node)) == 1
        self.name = name
        self.node = node

    def __str__(self):
        return self.__repr__()

    def __repr__(self):
        return self.name

    def __hash__(self):
        return hash((self.name, self.node))

    def __len__(self):
        return 1

    def __eq__(self, other):
        if isinstance(other, Domain):
            return self.__hash__() == other.__hash__()
        else:
            raise NotImplemented()

    def copy(self):
        return Domain(self.name, self.node)
