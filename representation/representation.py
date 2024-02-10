import abc

from tables.column import Column
from tables.bexp import Bexp
from tables.domain import Domain


class RepresentationStep(abc.ABC):
    def __init__(self, name):
        self.name = name

    @abc.abstractmethod
    def __repr__(self):
        pass

    @abc.abstractmethod
    def __str__(self):
        pass

    def invert(self):
        return self


class StartTraversal(RepresentationStep):
    def __init__(self, start_columns):
        super().__init__("STT")
        self.start_columns = start_columns

    def __repr__(self):
        return f"{self.name} <{self.start_columns}>"

    def __str__(self):
        return self.__repr__()


class EndTraversal(RepresentationStep):
    def __init__(self, start_columns: list[Domain], end_columns: list[Domain]):
        super().__init__("ENT")
        self.start_columns = start_columns
        self.end_columns = end_columns

    def __repr__(self):
        return f"{self.name} <{self.start_columns}, {self.end_columns}>"

    def __str__(self):
        return self.__repr__()


class Traverse(RepresentationStep):
    def __init__(self, edge, columns=None):
        super().__init__("TRV")
        self.edge = edge
        self.hidden_keys = edge.get_hidden_keys()
        self.start_node = edge.from_node
        self.end_node = edge.to_node
        if columns is None:
            self.columns = []
        else:
            self.columns = columns

    def __repr__(self):
        return f"{self.name} <{self.edge}, {self.hidden_keys}>"

    def __str__(self):
        return self.__repr__()

    def invert(self):
        from schema.edge import SchemaEdge

        edge = self.edge
        rev = SchemaEdge.invert(edge)
        return Traverse(rev)


class Cross(RepresentationStep):
    def __init__(self, node):
        super().__init__("CRS")
        self.node = node

    def __repr__(self):
        return f"{self.name} <{self.node}>"

    def __str__(self):
        return self.__repr__()


class Expand(RepresentationStep):
    def __init__(self, start_node, end_node, indices, hidden_keys):
        super().__init__("EXP")
        self.start_node = start_node
        self.end_node = end_node
        self.indices = indices
        self.hidden_keys = hidden_keys

    def __repr__(self):
        return f"{self.name} <{self.start_node}, {self.end_node}, {self.hidden_keys}>"

    def __str__(self):
        return self.__repr__()

    def invert(self):
        return Project(self.end_node, self.start_node, self.indices)


class Equate(RepresentationStep):
    def __init__(self, start_node, end_node):
        super().__init__("EQU")
        self.start_node = start_node
        self.end_node = end_node

    def __repr__(self):
        return f"{self.name} <{self.start_node}, {self.end_node}>"

    def __str__(self):
        return self.__repr__()


class Get(RepresentationStep):
    def __init__(self, columns):
        super().__init__("GET")
        self.columns = columns

    def __repr__(self):
        return f"{self.name} <{self.columns}>"

    def __str__(self):
        return self.__repr__()


class Push(RepresentationStep):
    def __init__(self):
        super().__init__("PSH")

    def __repr__(self):
        return self.name

    def __str__(self):
        return self.__repr__()


class Merge(RepresentationStep):
    def __init__(self):
        super().__init__("MER")

    def __repr__(self):
        return self.name

    def __str__(self):
        return self.__repr__()


class Pop(RepresentationStep):
    def __init__(self):
        super().__init__("POP")

    def __repr__(self):
        return self.name

    def __str__(self):
        return self.__repr__()


class Call(RepresentationStep):
    def __init__(self):
        super().__init__("CAL")

    def __repr__(self):
        return self.name

    def __str__(self):
        return self.__repr__()


class Return(RepresentationStep):
    def __init__(self):
        super().__init__("RET")

    def __repr__(self):
        return self.name

    def __str__(self):
        return self.__repr__()


class Reset(RepresentationStep):
    def __init__(self):
        super().__init__("RST")

    def __repr__(self):
        return self.name

    def __str__(self):
        return self.__repr__()


class Rename(RepresentationStep):
    def __init__(self, mapping):
        super().__init__("RNM")
        self.mapping = mapping

    def __repr__(self):
        return f"{self.name} <{self.mapping}>"

    def __str__(self):
        return self.__repr__()


class Project(RepresentationStep):
    def __init__(self, start_node, end_node, indices):
        super().__init__("PRJ")
        self.start_node = start_node
        self.end_node = end_node
        self.indices = indices

    def __repr__(self):
        return f"{self.name} <{self.start_node}, {self.end_node}, {self.indices}>"

    def __str__(self):
        return self.__repr__()

    def invert(self):
        from schema.node import SchemaNode

        nodes = SchemaNode.get_constituents(self.start_node)
        hidden_keys = [n for (i, n) in enumerate(nodes) if i not in set(self.indices)]
        return Expand(self.end_node, self.start_node, self.indices, hidden_keys)


class Drop(RepresentationStep):
    def __init__(self, columns):
        super().__init__("DRP")
        self.columns = columns

    def __repr__(self):
        return f"{self.name} <{self.columns}>"

    def __str__(self):
        return self.__repr__()


class Filter(RepresentationStep):
    def __init__(self, column):
        super().__init__("FLT")
        self.column = column

    def __repr__(self):
        return f"{self.name} <{self.column}>"

    def __str__(self):
        return self.__repr__()


class Sort(RepresentationStep):
    def __init__(self, columns: list[str]):
        super().__init__("SRT")
        self.columns = columns

    def __repr__(self):
        return f"{self.name} <{self.columns}>"

    def __str__(self):
        return self.__repr__()


class End(RepresentationStep):
    def __init__(self, left, hidden, right):
        super().__init__("END")
        self.left = left
        self.hidden = hidden
        self.right = right

    def __repr__(self):
        return f"{self.name} <{self.left}, {self.hidden}, {self.right}>"

    def __str__(self):
        return self.__repr__()
