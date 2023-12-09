import uuid

import numpy as np
import pandas as pd

from schema.node import SchemaNode
from tables.column import Column
import copy

from tables.derivation import DerivationStep, Rename, End, Project, Filter


class Table:
    def __init__(self, table_id, derivation: list[DerivationStep], schema, derived_from = None,):
        self.table_id = table_id
        self.derived_from = derived_from
        self.displayed_columns = []
        self.marker = 0 #index of the first value column
        self.columns = {}
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
        table.displayed_columns = [str(k) for k in keys]
        table.columns = {str(k): k for k in keys}
        table.marker = len(keys)
        table.execute()
        return table

    def execute(self):
        self.df = self.schema.execute_query(self.table_id, self.derived_from, self.derivation)

    @classmethod
    def create_from_table(cls, table):
        table_id = uuid.uuid4().hex
        new_table = Table(table_id,
                          copy.deepcopy(table.derivation),
                          copy.deepcopy(table.schema),
                          table.table_id)
        new_table.displayed_columns = copy.deepcopy(table.displayed_columns)
        new_table.marker = table.marker
        new_table.columns = copy.deepcopy(table.columns)
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

    def new_col_from_node(self, node: SchemaNode):
        name = self.get_fresh_name(str(node))
        if name != str(node):
            new_node = self.schema.clone(node, name)
        else:
            new_node = node
        return Column(name, new_node, [])

    def clone(self, column: Column):
        name = self.get_fresh_name(column.name)
        new_node = self.schema.clone(column.node, name)
        return Column(name, new_node, column.keyed_by)

    def compose(self, from_keys: list[str], to_key: str, via: list[str] = None):
        assert len(from_keys) == len(set(from_keys))
        key_idx = self.displayed_columns.index(to_key)
        assert 0 <= key_idx < self.marker
        key = self.columns[to_key]
        start_node = key.node
        via_nodes = None
        if via is not None:
            via_nodes = [self.schema.get_node_with_name(n) for n in via]
        end_nodes = [self.schema.get_node_with_name(c) for c in from_keys]
        end_node = SchemaNode.product(end_nodes)
        _, d, hidden_keys = self.schema.find_shortest_path(start_node, end_node, via_nodes, backwards=True)
        new_table = Table.create_from_table(self)
        new_cols = [new_table.new_col_from_node(node) for node in end_nodes]
        old_names = from_keys
        new_names = [str(c) for c in new_cols]
        new_table.namespace -= set(to_key)
        hidden_columns = [new_table.new_col_from_node(k) for k in hidden_keys]

        def compose_columns(column: Column):
            if key in set(column.keyed_by):
                new_key = [k for k in column.keyed_by if k != key] + new_cols + hidden_columns
                return Column(column.name, column.node, new_key)
            else:
                return column

        new_table.columns = ({c: compose_columns(self.columns[c]) for c in new_table.displayed_columns[:key_idx]}
                             | {new_table.displayed_columns[i+key_idx]: c for i, c in enumerate(new_cols)}
                             | {c: compose_columns(self.columns[c]) for c in new_table.displayed_columns[key_idx + len(new_cols):]}
                             | {str(col): col for col in hidden_columns})
        new_table.marker = self.marker + len(new_table.columns) - len(self.columns)
        keys = [new_table.columns[c] for c in new_table.displayed_columns[:new_table.marker]]
        vals = [new_table.columns[c] for c in new_table.displayed_columns[new_table.marker:]]
        new_derivation = d[:-1] + [Rename({old_name: name for old_name, name in zip(old_names, new_names)}), d[-1], End(keys, [], vals)]
        new_table.derivation = self.derivation[:-1] + new_derivation
        new_table.df = self.schema.execute_query(new_table.table_id, self.table_id, new_table.derivation)
        return new_table

    def infer(self, from_columns: list[str], to_column: str, via: list[str] = None, with_name: str = None):
        cols = [self.columns[fc] for fc in from_columns]
        nodes = [c.node for c in cols]
        start_node = SchemaNode.product(nodes)
        via_nodes = None
        if via is not None:
            via_nodes = [self.schema.get_node_with_name(n) for n in via]
        end_node = self.schema.get_node_with_name(to_column)
        _, d, hidden_keys = self.schema.find_shortest_path(start_node, end_node, via_nodes)
        new_table = Table.create_from_table(self)
        name = str(to_column)

        if with_name is not None:
            name = with_name
        old_name = name
        name = new_table.get_fresh_name(name)

        hidden_keys_raw_str = [str(hk) for hk in hidden_keys] # olc names
        hidden_keys_cols = [new_table.new_col_from_node(hk) if str(hk) != name else Column(name, hk, []) for hk in hidden_keys]
        hidden_keys_str = [str(c) for c in hidden_keys_cols]
        mapping = {old: new for old, new in zip(hidden_keys_raw_str, hidden_keys_str)}

        for h in hidden_keys_cols:
            new_table.columns[str(h)] = h

        new_col = Column(name, end_node, cols + hidden_keys_cols)
        new_table.columns[str(new_col)] = new_col

        keys = [new_table.columns[c] for c in new_table.displayed_columns[:new_table.marker]]
        vals = [new_table.columns[c] for c in new_table.displayed_columns[new_table.marker:]] + [new_col]

        new_col = Column(name, end_node, cols)

        new_table.displayed_columns += [str(new_col)]

        new_derivation = d[:-1] + [Rename({old_name: name} | mapping), d[-1], End(keys, hidden_keys_cols, vals)]

        new_table.derivation = self.derivation[:-1] + new_derivation
        new_table.df = self.schema.execute_query(new_table.table_id, self.table_id, new_table.derivation)
        new_table.schema = self.schema
        return new_table

    def get_column_from_index(self, index: int):
        if 0 <= index < len(self.displayed_columns):
            return self.columns[index]

    def combine(self, with_table):
        pass

    def hide(self, col):
        new_table = Table.create_from_table(self)
        idx = self.displayed_columns.index(col)
        assert 0 <= idx
        new_table.marker -= 1
        new_table.displayed_columns = self.displayed_columns[:idx] + self.displayed_columns[idx+1:]
        keys = [new_table.columns[c] for c in new_table.displayed_columns[:new_table.marker]]
        vals = [new_table.columns[c] for c in new_table.displayed_columns[new_table.marker:]]
        hidden_keys = list(set(new_table.columns.values()) - set(keys) - set(vals))
        new_table.derivation = self.derivation[:-1] + [End(keys, hidden_keys, vals)]
        new_table.df = new_table.schema.execute_query(new_table.table_id, self.table_id, new_table.derivation)
        return new_table

    def show(self, col):
        new_table = Table.create_from_table(self)
        assert col in self.columns.keys()
        column = self.columns[col]
        must_be_before = len(self.displayed_columns)-1
        must_be_after = 0
        for (i, x) in enumerate(self.displayed_columns):
            c = self.columns[x]
            if column in set(c.keyed_by):
                must_be_before = min(must_be_before, i)
            if c in set(column.keyed_by):
                must_be_after = max(must_be_after, i)
        if col in self.columns.keys():
            must_be_after -= 1
        assert must_be_after < must_be_before
        if must_be_after <= self.marker:
            must_be_after = min(must_be_before, new_table.marker)
            new_table.marker += 1

        new_table.displayed_columns.insert(must_be_after, col)

        keys = [new_table.columns[c] for c in new_table.displayed_columns[:new_table.marker]]
        vals = [new_table.columns[c] for c in new_table.displayed_columns[new_table.marker:]]
        hidden_keys = list(set(new_table.columns.values()) - set(keys) - set(vals))

        new_table.derivation = self.derivation[:-1] + [End(keys, hidden_keys, vals)]
        new_table.df = new_table.schema.execute_query(new_table.table_id, self.table_id, new_table.derivation)
        return new_table

    def filter(self, predicate):
        new_table = Table.create_from_table(self)
        new_table.derivation = self.derivation[:-1] + [Filter(predicate), self.derivation[-1]]
        new_table.df = new_table.schema.execute_query(new_table.table_id, self.table_id, new_table.derivation)
        return new_table


    def __repr__(self):
        keys = ' '.join([str(self.columns[k]) for k in self.displayed_columns[:self.marker]])
        vals = ' '.join([str(self.columns[v]) for v in self.displayed_columns[self.marker:]])
        return f"[{keys} || {vals}]" + "\n" + str(self.df)

    def __str__(self):
        return self.__repr__()

    def __getitem__(self, item):
        return self.columns[item]


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
