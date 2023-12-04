class Table:
    def __init__(self, keys, values, derivation):
        self.keys = keys
        self.values = values
        self.derivation = derivation

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