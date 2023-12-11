import copy
import typing
import uuid
from collections import deque

import pandas as pd

from schema import SchemaEdge, Cardinality, reverse_cardinality
from schema.node import SchemaNode
from tables.column import Column
from tables.exceptions import KeyMismatchException
from tables.function import Function
from tables.predicate import Predicate
from tables.raw_column import RawColumn, ColumnType
from tables.derivation import DerivationStep, Rename, End, Filter, StartTraversal, EndTraversal, Traverse


def find_index(l: list, v):
    try:
        return l.index(v)
    except ValueError:
        return -1


def invert_derivation(derivation: list[SchemaEdge]):
    new_derivation = []
    is_relational = False
    for i in range(len(derivation)-1, -1, -1):
        e = derivation[i]
        new_derivation += [SchemaEdge(e.to_node, e.from_node, reverse_cardinality(e.cardinality))]
        is_relational |= (reverse_cardinality(e.cardinality) == Cardinality.MANY_TO_MANY
                          or reverse_cardinality(e.cardinality) == Cardinality.ONE_TO_MANY)
    return is_relational, new_derivation

class Table:
    def __init__(self, table_id, intermediate_representation: list[DerivationStep], schema, derived_from=None, ):
        self.table_id = table_id
        self.derived_from = derived_from
        self.displayed_columns = []
        self.marker = 0  # index of the first value column
        self.keys = {}
        self.hidden_keys = {}
        self.values = {}
        self.intermediate_representation = intermediate_representation
        self.schema = schema
        self.namespace = set()
        self.df = pd.DataFrame()
        self.dropped_keys_count = 0
        self.dropped_vals_count = 0

    @classmethod
    def construct(cls, key_nodes: list[SchemaNode], intermediate_representation, schema):
        table_id = uuid.uuid4().hex
        table = Table(table_id, intermediate_representation, schema)
        keys = []
        for key_node in key_nodes:
            key = table.create_column(key_node, [], ColumnType.KEY)
            keys += [key]
        table.displayed_columns = [str(k) for k in keys]
        table.keys = {str(k): k for k in keys}
        table.dropped_keys_count = 0
        table.dropped_vals_count = 0
        table.marker = len(keys)
        table.execute()
        return table

    def execute(self):
        self.df, self.dropped_keys_count, self.dropped_vals_count = self.schema.execute_query(self.table_id, self.derived_from, self.intermediate_representation)


    @classmethod
    def create_from_table(cls, table):
        table_id = uuid.uuid4().hex
        new_table = Table(table_id,
                          copy.deepcopy(table.intermediate_representation),
                          copy.deepcopy(table.schema),
                          table.table_id)
        new_table.displayed_columns = copy.deepcopy(table.displayed_columns)
        new_table.marker = table.marker
        new_table.keys = copy.deepcopy(table.keys)
        new_table.values = copy.deepcopy(table.values)
        new_table.hidden_keys = copy.deepcopy(table.hidden_keys)
        new_table.namespace = copy.deepcopy(table.namespace)
        new_table.dropped_keys_count = table.dropped_keys_count
        new_table.dropped_vals_count = table.dropped_vals_count
        return new_table

    def create_column(self, node: SchemaNode, keys, type: ColumnType) -> RawColumn:
        constituents = SchemaNode.get_constituents(node)
        assert len(constituents) == 1
        c = constituents[0]
        name = self.get_fresh_name(str(c))
        return RawColumn(name, node, keys, type, [], self)

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

    def new_col_from_node(self, node: SchemaNode, type: ColumnType):
        name = self.get_fresh_name(str(node))
        if name != str(node):
            new_node = self.schema.clone(node, name)
        else:
            new_node = node
        return RawColumn(name, new_node, [], type)

    def clone(self, column: RawColumn):
        name = self.get_fresh_name(column.name)
        new_node = self.schema.clone(column.node, name)
        return RawColumn(name, new_node, column.keyed_by, column.type, [], self)

    def compose(self, from_keys: list[str], to_key: str, via: list[str] = None):
        assert len(from_keys) == len(set(from_keys))
        key_idx = find_index(self.displayed_columns, to_key)
        assert 0 <= key_idx < self.marker
        key = self.keys[to_key]
        start_node = key.node
        via_nodes = None
        if via is not None:
            via_nodes = [self.schema.get_node_with_name(n) for n in via]
        end_nodes = [self.schema.get_node_with_name(c) for c in from_keys]
        end_node = SchemaNode.product(end_nodes)
        derivation, d, hidden_keys = self.schema.find_shortest_path(start_node, end_node, via_nodes, backwards=True)
        new_table = Table.create_from_table(self)
        new_cols = [new_table.new_col_from_node(node, ColumnType.KEY) if str(node) != str(start_node) else start_node for node in end_nodes]
        old_names = from_keys
        new_names = [str(c) for c in new_cols]
        new_table.namespace -= set(to_key)
        hidden_columns = [new_table.new_col_from_node(k, ColumnType.KEY) for k in hidden_keys]

        for c in list(new_table.keys.values()) + list(new_table.values.values()) + list(new_table.hidden_keys.values()):
            c = typing.cast(RawColumn, c)
            if key in set(c.keyed_by):
                c.set_derivation(derivation + c.get_derivation())

        def compose_columns(column: RawColumn, column_type):
            if key in set(column.keyed_by):
                new_key = [k for k in column.keyed_by if k != key] + new_cols + hidden_columns
                return RawColumn(column.name, column.node, new_key, column_type, column.derivation, self)
            else:
                return column

        new_table.marker = self.marker + len(from_keys) - 1

        new_table.displayed_columns = self.displayed_columns[:key_idx] + [str(c) for c in new_cols] + self.displayed_columns[key_idx+1:]

        new_table.keys = ({c: compose_columns(self.keys[c], ColumnType.KEY) for c in new_table.displayed_columns[:key_idx]}
                          | {new_table.displayed_columns[i + key_idx]: c for i, c in enumerate(new_cols)}
                          | {c: compose_columns(self.keys[c], ColumnType.KEY) for c in
                             new_table.displayed_columns[key_idx + len(new_cols):new_table.marker]})
        new_table.hidden_keys = self.hidden_keys | {str(col): col for col in hidden_columns}
        new_table.values = {c: compose_columns(self.values[c], ColumnType.VALUE) for c in new_table.displayed_columns[new_table.marker:]}
        keys = [new_table.keys[c] for c in new_table.displayed_columns[:new_table.marker]]
        vals = [new_table.values[c] for c in new_table.displayed_columns[new_table.marker:]]

        new_intermediate_representation = d[:-1] + [Rename({old_name: name for old_name, name in zip(old_names, new_names)}), d[-1],
                                   End(keys, [], vals)]
        new_table.intermediate_representation = self.intermediate_representation[:-1] + new_intermediate_representation
        new_table.df, new_table.dropped_keys_count, new_table.dropped_vals_count = self.schema.execute_query(new_table.table_id, self.table_id, new_table.intermediate_representation)
        new_table.schema = self.schema
        return new_table

    # TODO: Handle the case where from columns is zero
    def infer(self, from_columns: list[str], to_column: str, via: list[str] = None, with_name: str = None):
        cols = [self.keys[fc] if fc in self.keys.keys() else self.values[fc] for fc in from_columns]
        nodes = [c.node for c in cols]
        start_node = SchemaNode.product(nodes)
        via_nodes = None
        if via is not None:
            via_nodes = [self.schema.get_node_with_name(n) for n in via]
        end_node = self.schema.get_node_with_name(to_column)
        derivation, representation, hidden_keys = self.schema.find_shortest_path(start_node, end_node, via_nodes)
        new_table = Table.create_from_table(self)
        name = str(to_column)

        if with_name is not None:
            name = with_name
        old_name = name
        name = new_table.get_fresh_name(name)

        hidden_keys_raw_str = [str(hk) for hk in hidden_keys]  # olc names
        hidden_keys_cols = [
            new_table.new_col_from_node(hk, ColumnType.KEY)
            if str(hk) != name
            else RawColumn(name, hk, [], ColumnType.KEY, [], self)
            for hk in hidden_keys]
        hidden_keys_str = [str(c) for c in hidden_keys_cols]
        mapping = {old: new for old, new in zip(hidden_keys_raw_str, hidden_keys_str)}

        for h in hidden_keys_cols:
            new_table.hidden_keys[str(h)] = h

        new_col = RawColumn(name, end_node, cols + hidden_keys_cols, ColumnType.VALUE, derivation, self)
        new_table.values[str(new_col)] = new_col

        keys = [new_table.keys[c] for c in new_table.displayed_columns[:new_table.marker]]
        vals = [new_table.values[c] for c in new_table.displayed_columns[new_table.marker:]] + [new_col]

        new_table.displayed_columns += [str(new_col)]

        new_intermediate_representation = representation[:-1] + [Rename({old_name: name} | mapping), representation[-1], End(keys, hidden_keys_cols, vals)]

        new_table.intermediate_representation = self.intermediate_representation[:-1] + new_intermediate_representation
        new_table.df, new_table.dropped_keys_count, new_table.dropped_vals_count = self.schema.execute_query(new_table.table_id, self.table_id, new_table.intermediate_representation)
        new_table.schema = self.schema
        return new_table

    def get_column_from_index(self, index: int):
        if 0 <= index < self.marker:
            return self.keys[index]
        if index < len(self.displayed_columns):
            return self.values[index]

    def combine(self, with_table):
        pass

    def hide(self, col):
        new_table = Table.create_from_table(self)
        idx = find_index(self.displayed_columns, col)
        assert 0 <= idx
        if idx < self.marker:
            new_table.marker -= 1
            to_hide = self.keys[col]
        else:
            to_hide = self.values[col]
        new_table.displayed_columns = self.displayed_columns[:idx] + self.displayed_columns[idx + 1:]
        keys = [new_table.keys[c] for c in new_table.displayed_columns[:new_table.marker]]
        vals = [new_table.values[c] for c in new_table.displayed_columns[new_table.marker:]]
        hidden_keys = list(set(self.hidden_keys.values())) + [to_hide]
        new_table.keys = {str(c): c for c in keys}
        new_table.values = {str(c): c for c in vals}
        new_table.hidden_keys = {str(c): c for c in hidden_keys}
        new_table.intermediate_representation = self.intermediate_representation[:-1] + [End(keys, hidden_keys, vals)]
        new_table.df, new_table.dropped_keys_count, new_table.dropped_vals_count = new_table.schema.execute_query(new_table.table_id, self.table_id, new_table.intermediate_representation)
        new_table.schema = self.schema
        return new_table

    def show(self, col):
        new_table = Table.create_from_table(self)
        assert col in self.hidden_keys.keys()
        column = self.hidden_keys[col]
        must_be_before = len(self.displayed_columns) - 1
        must_be_after = 0
        for (i, x) in enumerate(self.displayed_columns):
            if i < self.marker:
                c = self.keys[x]
            else:
                c = self.values[x]
            if column in set(c.keyed_by):
                must_be_before = min(must_be_before, i)
            if c in set(column.keyed_by):
                must_be_after = max(must_be_after, i)
        assert must_be_after < must_be_before
        idx = must_be_after
        if idx <= self.marker:
            idx = min(must_be_before, new_table.marker)
            if column.type == ColumnType.KEY or str(column) in self.displayed_columns:
                new_table.marker += 1
        new_table.displayed_columns.insert(idx, col)
        keys = [new_table.keys[c] if c in new_table.keys.keys() else column for c in new_table.displayed_columns[:new_table.marker]]
        vals = [new_table.values[c] if c in new_table.values.keys() else column for c in new_table.displayed_columns[new_table.marker:]]
        hidden_keys = list(set(self.hidden_keys.values()) - {column})
        new_table.keys = {str(c): c for c in keys}
        new_table.values = {str(c): c for c in vals}
        new_table.hidden_keys = {str(c): c for c in hidden_keys}
        new_table.intermediate_representation = self.intermediate_representation[:-1] + [End(keys, hidden_keys, vals)]
        new_table.df, new_table.dropped_keys_count, new_table.dropped_vals_count = new_table.schema.execute_query(new_table.table_id, self.table_id, new_table.intermediate_representation)
        new_table.schema = self.schema
        return new_table

    def filter(self, predicate: Predicate):
        new_table = Table.create_from_table(self)
        new_table.intermediate_representation = self.intermediate_representation[:-1] + [Filter(predicate), self.intermediate_representation[-1]]
        new_table.df, new_table.dropped_keys_count, new_table.dropped_vals_count = new_table.schema.execute_query(new_table.table_id, self.table_id, new_table.intermediate_representation)
        new_table.schema = self.schema
        return new_table

    def set_key(self, key_list: list[str]):
        new_keys = {}
        new_displayed_columns = []
        keys_to_delete = set()
        vals_to_delete = set()
        derived_from_set = set()
        derived_from = {}
        derived_from_count = {}
        for (i, k) in enumerate(key_list):
            assert k in self.displayed_columns
            if k in self.keys.keys():
                col = self.keys[k]
                keys_to_delete.add(str(k))
            elif k in self.values.keys():
                col = self.values[k]
                vals_to_delete.add(str(k))
            else:
                raise Exception()
            keyed_by = [c for c in col.keyed_by if str(c) not in set(key_list[:i])]
            new_keys[str(k)] = RawColumn(col.name, col.node, [], ColumnType.KEY, col.derivation, self)
            new_displayed_columns += [str(k)]
            derived_f = self.values[k].keyed_by if k in self.values.keys() else self.keys[k].keyed_by
            to_explore = deque(derived_f)
            while len(to_explore) > 0:
                df = to_explore.popleft()
                if df not in derived_from_set or derived_from_count[df] > len(derived_f):
                    derived_from[df] = self.values[k]
                    derived_from_set.add(df)
                    derived_from_count[df] = len(derived_f)
                kb = df.keyed_by
                to_explore.extend(kb)
        derived_from_list = []
        others = []
        for i, col in enumerate(self.displayed_columns):
            if col in key_list:
                continue
            if i < self.marker:
                if col in keys_to_delete:
                    pass
                column = self.keys[col]
            else:
                if col in vals_to_delete:
                    pass
                column = self.values[col]
            if column in derived_from.keys():
                derived_from_list += [column]
            else:
                others += [column]
            new_displayed_columns += [col]
        new_values = {}
        new_hidden_keys = {}

        for val in others:
            new_values[str(val)] = RawColumn(val.name, val.node, val.keyed_by, ColumnType.VALUE, val.derivation, self)

        for val in derived_from_list:
            is_relational, new_derivation = invert_derivation(derived_from[val].derivation)
            if is_relational:
                if str(val) in set([str(v) for v in val.keyed_by]):
                    keyed_by = val.keyed_by
                else:
                    keyed_by = val.keyed_by + [val]
                new_values[str(val)] = RawColumn(val.name, val.node, keyed_by, ColumnType.VALUE, new_derivation, self)
                new_hidden_keys[str(val)] = RawColumn(val.name, val.node, [], ColumnType.VALUE, [], self)
            else:
                new_values[str(val)] = RawColumn(val.name, val.node, val.keyed_by, ColumnType.VALUE, new_derivation, self)

        new_table = Table.create_from_table(self)
        new_table.keys = new_keys
        new_table.values = new_values
        new_table.hidden_keys = self.hidden_keys | new_hidden_keys
        new_table.displayed_columns = new_displayed_columns

        new_table.marker = len(key_list)

        keys = [new_table.keys[c] for c in new_table.displayed_columns[:new_table.marker]]
        vals = [new_table.values[c] for c in new_table.displayed_columns[new_table.marker:]]
        hidden_keys = list(set(new_table.hidden_keys.values()))

        new_table.intermediate_representation = self.intermediate_representation[:-1] + [End(keys, hidden_keys, vals)]
        new_table.df, new_table.dropped_keys_count, new_table.dropped_vals_count = new_table.schema.execute_query(new_table.table_id, self.table_id,
                                                      new_table.intermediate_representation)
        new_table.schema = self.schema
        return new_table

    def equate(self, col1, col2):
        assert 0 <= self.displayed_columns.index(col1)
        assert 0 <= self.displayed_columns.index(col2)
        new_table = Table.create_from_table(self)
        idx = self.displayed_columns.index(col2)
        new_table.displayed_columns = [c for c in self.displayed_columns if c != col2]
        if idx < self.marker:
            new_table.marker -= 1
            column2 = self.keys[col2]
        else:
            column2 = self.values[col2]
        if self.displayed_columns.index(col1) < self.marker:
            column1 = self.keys[col1]
        else:
            column1 = self.values[col1]

        for col in list(self.keys.values()):
            if col == column2:
                continue
            keyed_by = [c if c != column2 else column1 for c in col.keyed_by]
            new_table.keys[str(col)] = RawColumn(col.name, col.node, keyed_by, ColumnType.KEY, col.derivation, self)

        for col in list(self.values.values()):
            if col == column2:
                continue
            keyed_by = [c if c != column2 else column1 for c in col.keyed_by]
            new_table.values[str(col)] = RawColumn(col.name, col.node, keyed_by, ColumnType.VALUE, col.derivation, self)

        for col in list(self.values.values()):
            if col == column2:
                continue
            keyed_by = [c if c != column2 else column1 for c in col.keyed_by]
            new_table.hidden_keys[str(col)] = RawColumn(col.name, col.node, keyed_by, col.type, col.derivation, self)

        keys = [new_table.keys[c] for c in new_table.displayed_columns[:new_table.marker]]
        vals = [new_table.values[c] for c in new_table.displayed_columns[new_table.marker:]]
        hidden_keys = list(set(self.hidden_keys.values()))

        new_table.intermediate_representation = self.intermediate_representation[:-1] + [Filter(Column(column1) == Column(column2)), End(keys, hidden_keys, vals)]
        new_table.df, new_table.dropped_keys_count, new_table.dropped_vals_count = new_table.schema.execute_query(new_table.table_id, self.table_id,
                                                      new_table.intermediate_representation)
        new_table.schema = self.schema
        return new_table

    def assign(self, name: str, function: Function):
        name = self.get_fresh_name(name)
        arguments = function.arguments
        columns = [a for a in arguments if isinstance(a, Column)]
        columns_dedup = [c.raw_column for c in set(columns)]
        nodes = [c.raw_column.node for c in columns]
        start_node = SchemaNode.product(nodes)
        hidden_keys = [c.get_hidden_keys() for c in columns]
        shk = set()
        for hk in hidden_keys:
            if shk.issubset(hk) or hk.issubset(shk):
                shk.union(hk)
            else:
                raise KeyMismatchException(shk, hk)
        end_node = self.schema.add_node(name)
        edge = self.schema.add_edge(start_node, end_node, function.cardinality)
        self.schema.map_edge_to_closure_function(edge, function)

        new_table = Table.create_from_table(self)
        column = RawColumn(name, end_node, columns_dedup, ColumnType.VALUE, [edge], new_table)
        new_table.displayed_columns += [str(column)]
        new_table.values = self.values | {str(column): column}
        keys = [new_table.keys[c] for c in new_table.displayed_columns[:new_table.marker]]
        vals = [new_table.values[c] for c in new_table.displayed_columns[new_table.marker:]]

        traversal = [StartTraversal(start_node, Traverse(start_node, end_node)), EndTraversal(start_node, end_node)]

        new_table.intermediate_representation = self.intermediate_representation[:-1] + traversal + [End(keys, hidden_keys, vals)]
        new_table.df, new_table.dropped_keys_count, new_table.dropped_vals_count = new_table.schema.execute_query(new_table.table_id, self.table_id,
                                                      new_table.intermediate_representation)
        new_table.schema = self.schema
        return new_table

    def __repr__(self):
        keys = ' '.join([str(self.keys[k]) for k in self.displayed_columns[:self.marker]])
        vals = ' '.join([str(self.values[v]) for v in self.displayed_columns[self.marker:]])
        dropped_keys = f"\n{self.dropped_keys_count} keys hidden"
        dropped_vals = f"\n{self.dropped_vals_count} values hidden"
        repr = f"[{keys} || {vals}]" + "\n" + str(self.df)
        if self.dropped_keys_count > 0:
            repr += dropped_keys
        if self.dropped_vals_count > 0:
            repr += dropped_vals
        return repr + "\n\n"

    def __str__(self):
        return self.__repr__()

    def __getitem__(self, item):
        return Column(self.keys[item]) if item in self.keys.keys() else Column(self.values[item])

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
