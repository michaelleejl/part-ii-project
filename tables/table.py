import uuid

from schema.node import SchemaNode
from tables.column import Column
import copy

from tables.derivation import DerivationStep


class Table:
    def __init__(self, table_id, derivation: list[DerivationStep], schema):
        self.table_id = table_id
        self.columns = []
        self.marker = 0 #index of the first value column
        self.keys = {}
        self.values = {}
        self.derivation = derivation
        self.schema = schema
        self.namespace = set()

    @classmethod
    def construct(cls, key_nodes: list[SchemaNode], derivation, schema):
        table_id = uuid.uuid4().hex
        table = Table(table_id, derivation, schema)
        keys = []
        for key_node in key_nodes:
            key = table.create_column(key_node, [])
            keys += [key]
        table.columns = [str(k) for k in keys]
        table.keys = {i: keys[i] for i in range(len(keys))}
        table.marker = len(keys)
        return table

    @classmethod
    def create_from_table(cls, table):
        new_table = Table(table.table_id, copy.deepcopy(table.derivation), copy.deepcopy(table.schema))
        new_table.columns = copy.deepcopy(table.columns)
        new_table.marker = table.marker
        new_table.keys = copy.deepcopy(table.keys)
        new_table.values = copy.deepcopy(table.values)
        new_table.namespace = copy.deepcopy(table.namespace)
        return new_table

    def create_column(self, node: SchemaNode, keys) -> Column:
        constituents = SchemaNode.get_constituents(node)
        assert len(constituents) == 1
        c = constituents[0]
        name = self.get_fresh_name(str(c))
        return Column(name, node, keys)

    def get_fresh_name(self, name: str):
        candidate = name
        if candidate in self.namespace:
            i = 1
            candidate = f"{name}_{i}"
            while candidate in self.namespace:
                i += 1
                candidate = f"{name}_{i}"
        self.namespace.add(candidate)
        return candidate

    def clone(self, column: Column):
        name = self.get_fresh_name(column.name)
        return Column(name, column.node, column.keyed_by)

    def compose(self, with_edge):
        pass

    def infer(self, from_columns: list[str], to_column: str, via: list[str] = None, with_name: str = None):
        cols_idx = [self.columns.index(c) for c in from_columns]
        cols = [self.get_column_from_index(idx) for idx in cols_idx]
        nodes = [c.node for c in cols]
        start_node = SchemaNode.product(nodes)
        via_nodes = None
        if via is not None:
            via_nodes = [self.schema.get_node_with_name(n) for n in via]
        end_node = self.schema.get_node_with_name(to_column)
        self.schema.find_shortest_path(start_node, end_node, via_nodes)
        name = str(to_column)
        if with_name is not None:
            name = with_name
        name = self.get_fresh_name(name)
        new_col = Column(name, end_node, cols_idx)
        new_table = Table.create_from_table(self)
        new_table.columns += [str(new_col)]
        new_table.values[len(self.columns)] = new_col
        return new_table

    def get_column_from_index(self, index: int):
        if 0 <= index < self.marker:
            return self.keys[index]
        elif self.marker <= index < len(self.columns):
            return self.values[index]

    def combine(self, with_table):
        pass

    def hide(self, key):
        pass

    def show(self, key):
        pass

    def make_value(self, node):
        pass

    def __repr__(self):
        keys = ' '.join([str(self.keys[k]) for k in range(self.marker)])
        vals = ' '.join([str(self.values[v]) for v in range(self.marker, len(self.columns))])
        return f"[{keys} || {vals}]"

    def __str__(self):
        return self.__repr__()


    ## the task is
    ## given a schema where (bank | cardnum) and (bonus | cardnum, person)
    ## I want [cardnum person || bank bonus] as the values

    ## t = schema.get([cardnum, person]) [cardnum person || unit]
    ## 2 possibilities: cardnum x person or the specific cardnum, person pairs that key bonus.

    ## t = t.infer([cardnum, person] -> cardnum)
    ## t = t.infer(cardnum -> bank).add_value(bank)
    ## t = t.infer([cardnum, person] -> bonus).add_value(bonus)



    ## Example of composition
    ## Schema: Order -> payment method -> billing address
    ## Goal is: [order || billing address]

    ## I can do
    ## t = schema.get([payment method]) [payment method || unit]

    ## t = t.compose(order -> payment method)
    ## t = t.infer(payment method -> billing address).add_value(billing address) [order || billing address]
    ## t = t.infer(billing address -> shipping fee)

    ## Order
    ##  |
    ## payment method
    ##  |
    ## billing address

    ## turn them into test suites
