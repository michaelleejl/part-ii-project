import uuid

import numpy as np
import pandas as pd

from schema.node import SchemaNode
from tables.column import Column
import copy

from tables.derivation import DerivationStep, Rename, End


class Table:
    def __init__(self, table_id, derivation: list[DerivationStep], schema, derived_from = None,):
        self.table_id = table_id
        self.derived_from = derived_from
        self.columns = []
        self.marker = 0 #index of the first value column
        self.keys = {}
        self.values = {}
        self.derivation = derivation
        self.schema = schema
        self.namespace = set()
        self.df = pd.DataFrame()

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
        table.execute()
        return table

    def execute(self):
        print(self.derivation)
        self.df = self.schema.execute_query(self.table_id, self.derived_from, self.derivation)

    @classmethod
    def create_from_table(cls, table):
        table_id = uuid.uuid4().hex
        new_table = Table(table_id,
                          copy.deepcopy(table.derivation),
                          copy.deepcopy(table.schema),
                          table.table_id)
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
        new_node = self.schema.clone(column.node, name)
        return Column(name, new_node, column.keyed_by)

    def compose(self, from_keys: list[str], to_keys: list[str], via: list[str] = None):
        assert len(from_keys) == len(set(from_keys))
        keys_idx = [self.columns.index(c) for c in to_keys]
        min_idx = min(keys_idx)
        assert np.all(0 <= np.array(keys_idx) < self.marker)
        keys = [self.get_column_from_index(idx) for idx in keys_idx]
        nodes = [c.node for c in keys]
        start_node = SchemaNode.product(nodes)
        via_nodes = None
        if via is not None:
            via_nodes = [self.schema.get_node_with_name(n) for n in via]
        end_nodes = [self.schema.get_node_with_name(c) for c in from_keys]
        end_node = SchemaNode.product(end_nodes)
        _, d, _ = self.schema.find_shortest_path(start_node, end_node, via_nodes)
        old_names = from_keys
        new_table = Table.create_from_table(self)
        names = [new_table.get_fresh_name(name) for name in old_names]
        keys_str = [str(new_table.keys[i]) for i in range(new_table.marker) if str(new_table.keys[i]) not in set(to_keys)]
        vals_str = [str(new_table.values[i]) for i in range(new_table.marker, len(new_table.columns))]
        keys_str = keys_str[:min_idx] + names + keys_str[min_idx:]
        new_derivation = d[:-1] + [Rename({old_name: name for old_name, name in zip(old_names, names)}), d[-1], End(keys_str, vals_str)]
        new_table.keys = ({i: self.keys[i] for i in range(min_idx)}
                          | {min_idx+i: Column(names[i], end_nodes[i], []) for i in range(len(names))}
                          | {i+len(names): self.keys[i] for i in range(min_idx + len(names), len(keys_str))})
        new_table.marker = self.marker + len(new_table.keys) - len(self.keys)
        new_table.derivation = self.derivation[:-1] + new_derivation
        new_table.df = self.schema.execute_query(new_table.table_id, self.table_id, new_table.derivation)
        return new_table

    def infer(self, from_columns: list[str], to_column: str, via: list[str] = None, with_name: str = None):
        cols_idx = [self.columns.index(c) for c in from_columns]
        cols = [self.get_column_from_index(idx) for idx in cols_idx]
        nodes = [c.node for c in cols]
        start_node = SchemaNode.product(nodes)
        via_nodes = None
        if via is not None:
            via_nodes = [self.schema.get_node_with_name(n) for n in via]
        end_node = self.schema.get_node_with_name(to_column)
        _, d, _ = self.schema.find_shortest_path(start_node, end_node, via_nodes)
        name = str(to_column)
        if with_name is not None:
            name = with_name
        old_name = name
        new_table = Table.create_from_table(self)
        name = new_table.get_fresh_name(name)
        keys_str = [str(new_table.keys[i]) for i in range(new_table.marker)]
        vals_str = [str(new_table.values[i]) for i in range(new_table.marker, len(new_table.columns))] + [name]
        new_derivation = d[:-1] + [Rename({old_name: name}), d[-1], End(keys_str, vals_str)]
        new_col = Column(name, end_node, cols_idx)
        new_table.columns += [str(new_col)]
        new_table.values[len(new_table.columns) - 1] = new_col
        new_table.derivation = self.derivation[:-1] + new_derivation
        new_table.df = self.schema.execute_query(new_table.table_id, self.table_id, new_table.derivation)
        new_table.schema = self.schema
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
        return f"[{keys} || {vals}]" + "\n" + str(self.df)

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
