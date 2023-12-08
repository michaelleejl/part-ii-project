class DerivationStep:
    def __init__(self, type):
        self.type = type


class Traverse(DerivationStep):
    def __init__(self):
        super().__init__("TRAVERSE")


class Get(DerivationStep):
    def __init__(self, columns):
        super().__init__("GET")
        self.columns = columns


class Rename(DerivationStep):
    def __init__(self, mapping):
        super().__init__("RENAME")
        self.mapping = mapping


class Project(DerivationStep):
    def __init__(self, columns):
        super().__init__("PROJECT")
        self.columns = columns


class Filter(DerivationStep):
    def __init__(self, predicate):
        super().__init__("FILTER")
        self.predicate = predicate


class End(DerivationStep):
    def __init__(self, keys, values):
        super().__init__("END")
        self.keys = keys
        self.values = values
