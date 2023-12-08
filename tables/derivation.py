class DerivationStep:
    def __init__(self, name):
        self.name = name


class StartTraversal(DerivationStep):
    def __init__(self, step):
        super().__init__("STT")
        self.step = step

    def __repr__(self):
        return f"{self.name} <{self.step}>"

    def __str__(self):
        return self.__repr__()


class EndTraversal(DerivationStep):
    def __init__(self, start_node, end_node):
        super().__init__("ENT")
        self.start_node = start_node
        self.end_node = end_node

    def __repr__(self):
        return f"{self.name} <{self.start_node}, {self.end_node}>"

    def __str__(self):
        return self.__repr__()


class Traverse(DerivationStep):
    def __init__(self, start_node, end_node):
        super().__init__("TRV")
        self.start_node = start_node
        self.end_node = end_node

    def __repr__(self):
        return f"{self.name} <{self.start_node}, {self.end_node}>"

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
    def __init__(self, nodes):
        super().__init__("GET")
        self.nodes = nodes

    def __repr__(self):
        return f"{self.name} <{self.nodes}>"

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
    def __init__(self, columns):
        super().__init__("PRJ")
        self.columns = columns

    def __repr__(self):
        return f"{self.name} <{self.columns}>"

    def __str__(self):
        return self.__repr__()


class Filter(DerivationStep):
    def __init__(self, predicate):
        super().__init__("FLT")
        self.predicate = predicate

    def __repr__(self):
        return f"{self.name} <{self.predicate}>"

    def __str__(self):
        return self.__repr__()


class End(DerivationStep):
    def __init__(self, keys, values):
        super().__init__("END")
        self.keys = keys
        self.values = values

    def __repr__(self):
        return f"{self.name} <{self.keys}, {self.values}>"

    def __str__(self):
        return self.__repr__()
