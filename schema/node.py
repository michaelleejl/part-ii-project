import pandas as pd


class SchemaNode:
    def __init__(self, name, data, family):
        self.name = name
        self.family = family
        self.key = (self.name, self.family)
        df = pd.DataFrame(data)
        self.data = df.rename({c: f"{hash(self.key)}_{c}" for c in df.columns}, axis=1).reset_index()
        if len(df.columns) == 1:
            self.values = self.data[f"{hash(self.key)}_{df.columns[0]}"].values
        else:
            self.values = data.itertuples(index=False)

    @classmethod
    def update_data(cls, node, data: pd.DataFrame):
        return SchemaNode(node.name, data, node.family)

    def get_values(self):
        return self.values

    def get_unique_col_name(self, col_name: str) -> str:
        if f"{hash(self.key)}_{col_name}" in self.data.columns:
            return f"{hash(self.key)}_{col_name}"

    def get_key(self):
        return self.key

    def __hash__(self):
        return hash(self.get_key())

    def __eq__(self, other):
        if isinstance(other, SchemaNode):
            return self.get_key() == other.get_key()
        return NotImplemented

    def __repr__(self):
        return f"{self.name}: {len(self.data)} entries"

    def __str__(self):
        return f"{self.name} [{self.family}]"