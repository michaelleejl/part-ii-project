from tables.column import Column
from tables.predicate import Predicate
from tables.raw_column import RawColumn


class DerivationStep:
    def __init__(self, name):
        self.name = name


class StartTraversal(DerivationStep):
    def __init__(self, start_columns, explicit_keys):
        super().__init__("STT")
        self.start_columns = start_columns
        self.explicit_keys = explicit_keys

    def __repr__(self):
        return f"{self.name} <{self.start_columns}>"

    def __str__(self):
        return self.__repr__()


class EndTraversal(DerivationStep):
    def __init__(self, start_columns: list[RawColumn], end_columns: list[RawColumn]):
        super().__init__("ENT")
        self.start_columns = start_columns
        self.end_columns = end_columns

    def __repr__(self):
        return f"{self.name} <{self.start_columns}, {self.end_columns}>"

    def __str__(self):
        return self.__repr__()


class Traverse(DerivationStep):
    def __init__(self, start_node, end_node, hidden_keys=None, columns = None):
        super().__init__("TRV")
        self.start_node = start_node
        self.end_node = end_node
        if hidden_keys is None:
            self.hidden_keys = []
        else:
            self.hidden_keys = hidden_keys
        if columns is None:
            self.columns = []
        else:
            self.columns = []

    def __repr__(self):
        return f"{self.name} <{self.start_node}, {self.end_node}, {self.hidden_keys}, {self.columns}>"

    def __str__(self):
        return self.__repr__()


class Cross(DerivationStep):
    def __init__(self, node):
        super().__init__("CRS")
        self.node = node

    def __repr__(self):
        return f"{self.name} <{self.node}>"

    def __str__(self):
        return self.__repr__()


class Expand(DerivationStep):
    def __init__(self, start_node, end_node, hidden_keys=None, columns=None):
        super().__init__("EXP")
        self.start_node = start_node
        self.end_node = end_node
        if hidden_keys is None:
            self.hidden_keys = []
        else:
            self.hidden_keys = hidden_keys
        if columns is None:
            self.columns = []
        else:
            self.columns = []

    def __repr__(self):
        return f"{self.name} <{self.start_node}, {self.end_node}, {self.hidden_keys}, {self.columns}>"

    def __str__(self):
        return self.__repr__()


class Equate(DerivationStep):
    def __init__(self, start_node, end_node):
        super().__init__("EQU")
        self.start_node = start_node
        self.end_node = end_node

    def __repr__(self):
        return f"{self.name} <{self.start_node}, {self.end_node}>"

    def __str__(self):
        return self.__repr__()


class Get(DerivationStep):
    def __init__(self, columns):
        super().__init__("GET")
        self.columns = columns

    def __repr__(self):
        return f"{self.name} <{self.columns}>"

    def __str__(self):
        return self.__repr__()


class Rename(DerivationStep):
    def __init__(self, mapping):
        super().__init__("RNM")
        self.mapping = mapping

    def __repr__(self):
        return f"{self.name} <{self.mapping}>"

    def __str__(self):
        return self.__repr__()


class Project(DerivationStep):
    def __init__(self, start_node, end_node, hidden_keys=None, columns=None):
        super().__init__("PRJ")
        self.start_node = start_node
        self.end_node = end_node
        if hidden_keys is None:
            self.hidden_keys = []
        else:
            self.hidden_keys = hidden_keys
        if columns is None:
            self.columns = []
        else:
            self.columns = []

    def __repr__(self):
        return f"{self.name} <{self.start_node}, {self.end_node}, {self.hidden_keys}, {self.columns}>"

    def __str__(self):
        return self.__repr__()


class Filter(DerivationStep):
    def __init__(self, predicate: Predicate):
        super().__init__("FLT")
        self.predicate = predicate

    def __repr__(self):
        return f"{self.name} <{self.predicate}>"

    def __str__(self):
        return self.__repr__()


class Sort(DerivationStep):
    def __init__(self, columns: list[str]):
        super().__init__("SRT")
        self.columns = columns

    def __repr__(self):
        return f"{self.name} <{self.columns}>"

    def __str__(self):
        return self.__repr__()


class End(DerivationStep):
    def __init__(self, keys, hidden_keys, values):
        super().__init__("END")
        self.keys = keys
        self.hidden_keys = hidden_keys
        self.values = values

    def __repr__(self):
        return f"{self.name} <{self.keys}, {self.hidden_keys}, {self.values}>"

    def __str__(self):
        return self.__repr__()
