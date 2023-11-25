import pandas as pd


class SchemaNode:
    def __init__(self, name, family, clone_type=None):
        self.name = name
        self.family = family
        self.key = (self.name, self.family)
        self.clone_type = clone_type

    def prepend_id(self, val: str) -> str:
        return f"{hash(self.key)}_{val}"

    def get_key(self):
        return self.key

    def __hash__(self):
        return hash(self.get_key())

    def __eq__(self, other):
        if isinstance(other, SchemaNode):
            return self.get_key() == other.get_key()
        return NotImplemented

    def __repr__(self):
        return f"{self.name}"

    def __str__(self):
        return f"{self.name} [{self.family}]"