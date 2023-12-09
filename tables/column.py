from schema import SchemaNode


class Column:
    def __init__(self, name: str, node: SchemaNode, keyed_by: list[any]):
        self.name = name
        self.node = node
        self.keyed_by = keyed_by

    def __str__(self):
        return self.__repr__()

    def __repr__(self):
        return self.name

    def __hash__(self):
        return hash(self.name)

    def __eq__(self, other):
        if isinstance(other, Column):
            return self.name == other.name and self.keyed_by == other.keyed_by
        else:
            raise NotImplemented()
