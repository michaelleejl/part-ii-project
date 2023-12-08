from schema import SchemaNode


class Column:
    def __init__(self, name: str, node: SchemaNode, keyed_by):
        self.name = name
        self.node = node
        self.keyed_by = keyed_by

    def __str__(self):
        return self.__repr__()

    def __repr__(self):
        return self.name
