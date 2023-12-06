class Table:
    def __init__(self, keys, values, derivation, schema):
        self.keys = keys
        self.values = values
        self.derivation = derivation
        self.schema = schema

    def compose(self, with_edge):
        pass

    def infer(self, with_edge):
        pass

    def combine(self, with_table):
        pass

    def hide(self, key):
        pass

    def show(self, key):
        pass

    def __repr__(self):
        return f"[{' '.join([str(k) for k in self.keys])} || {' '.join([str(v) for v in self.values])}]"

    def __str__(self):
        return self.__repr__()