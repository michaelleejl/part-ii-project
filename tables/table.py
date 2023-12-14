import copy
import typing
import uuid
from collections import deque
from enum import Enum

import pandas as pd

from schema import SchemaEdge, Cardinality, reverse_cardinality
from schema.node import SchemaNode
from tables.aggregation import AggregationFunction
from tables.column import Column
from tables.exceptions import KeyMismatchException, ColumnsNeedToBeUniqueException, \
    ColumnsNeedToBeInTableAndVisibleException, ColumnsNeedToBeKeysException, ColumnsNeedToBeInTableException, \
    ColumnsNeedToBeHiddenException, IntermediateRepresentationMustHaveEndMarkerException
from tables.function import Function
from tables.predicate import Predicate
from tables.raw_column import RawColumn, ColumnType
from tables.derivation import DerivationStep, Rename, End, Filter, StartTraversal, EndTraversal, Traverse, Sort, \
    Project, Get


def find_index(l: list, v):
    try:
        return l.index(v)
    except ValueError:
        return -1


def invert_derivation(derivation: list[SchemaEdge]):
    new_derivation = []
    is_relational = False
    for i in range(len(derivation) - 1, -1, -1):
        e = derivation[i]
        new_derivation += [SchemaEdge(e.to_node, e.from_node, reverse_cardinality(e.cardinality))]
        is_relational |= (reverse_cardinality(e.cardinality) == Cardinality.MANY_TO_MANY
                          or reverse_cardinality(e.cardinality) == Cardinality.ONE_TO_MANY)
    return is_relational, new_derivation


def check_for_uniqueness(column_names):
    if len(column_names) != len(set(column_names)):
        raise ColumnsNeedToBeUniqueException()


def replace_key_and_update_derivation(key_to_be_replaced, key_to_be_replaced_by, new_derivation):
    def replacement_fun(column: RawColumn):
        if key_to_be_replaced in set(column.keyed_by):
            new_key = [k for k in column.keyed_by if k != key_to_be_replaced] + key_to_be_replaced_by
            return RawColumn(column.name, column.node, new_key, column.type, new_derivation + column.derivation,
                             column.table)
        else:
            return column

    return replacement_fun


class Table:
    class ColumnRequirements(Enum):
        IS_KEY = 1
        IS_HIDDEN = 2
        IS_VAL = 3
        IS_KEY_OR_VAL = 4
        IS_KEY_OR_HIDDEN = 5
        IS_VAL_OR_HIDDEN = 6
        IS_KEY_OR_VAL_OR_HIDDEN = 7
        IS_UNIQUE = 8

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
    def construct(cls, columns: list[tuple[SchemaNode, str]], schema):
        table_id = uuid.uuid4().hex
        table = Table(table_id, [], schema)
        keys = []
        for key_node, name in columns:
            key = table.create_column(name, key_node, [], ColumnType.KEY)
            keys += [key]

        table.displayed_columns = [str(k) for k in keys]
        derivation = [Get(keys), End(table.displayed_columns, [], [])]
        table.extend_intermediate_representation(derivation)
        table.keys = {str(k): k for k in keys}
        table.dropped_keys_count = 0
        table.dropped_vals_count = 0
        table.marker = len(keys)
        table.execute()
        return table

    def extend_intermediate_representation(self, with_new_representation: list[DerivationStep]):
        if not isinstance(with_new_representation[-1], End):
            raise IntermediateRepresentationMustHaveEndMarkerException()
        if len(self.intermediate_representation) > 0:
            self.intermediate_representation = self.intermediate_representation[:-1] + with_new_representation
        else:
            self.intermediate_representation = with_new_representation

    def execute(self):
        self.df, self.dropped_keys_count, self.dropped_vals_count, self.schema = self.schema.execute_query(
            self.table_id,
            self.derived_from,
            self.intermediate_representation)

    @classmethod
    def create_from_table(cls, table):
        table_id = uuid.uuid4().hex
        new_table = Table(table_id,
                          table.intermediate_representation,
                          table.schema,
                          table.table_id)
        new_table.displayed_columns = copy.copy(table.displayed_columns)
        new_table.marker = table.marker
        new_table.keys = copy.copy({k: RawColumn.assign_new_table(v, new_table) for k, v in table.keys.items()})
        new_table.values = copy.copy({k: RawColumn.assign_new_table(v, new_table) for k, v in table.values.items()})
        new_table.hidden_keys = copy.copy(
            {k: RawColumn.assign_new_table(v, new_table) for k, v in table.hidden_keys.items()})
        new_table.namespace = copy.deepcopy(table.namespace)
        new_table.dropped_keys_count = table.dropped_keys_count
        new_table.dropped_vals_count = table.dropped_vals_count
        return new_table

    def create_column(self, name, node: SchemaNode, keys, type: ColumnType) -> RawColumn:
        constituents = SchemaNode.get_constituents(node)
        assert len(constituents) == 1
        name = self.get_fresh_name(name)
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

    def new_col_from_node(self, node: SchemaNode, type: ColumnType, name: str = None):
        if name is None:
            name = self.get_fresh_name(str(node))
        else:
            name = self.get_fresh_name(name)
        original_name = str(node)
        if name != original_name:
            if len(name.split(".")) > 1:
                name = name.split(".")[1]
            new_node = self.schema.clone(node, name)
        else:
            new_node = node
        return RawColumn(str(new_node), new_node, [], type, table=self)

    def verify_columns(self, column_names: list[str], requirements: set[ColumnRequirements]):
        keys = set(self.keys.keys())
        vals = set(self.values.keys())
        hids = set(self.hidden_keys.keys())
        if Table.ColumnRequirements.IS_UNIQUE in requirements and len(set(column_names)) != len(column_names):
            raise ColumnsNeedToBeUniqueException()
        if Table.ColumnRequirements.IS_KEY_OR_VAL_OR_HIDDEN in requirements and not set(column_names).issubset(
                keys | vals | hids):
            raise ColumnsNeedToBeInTableException()
        if Table.ColumnRequirements.IS_KEY_OR_VAL in requirements and not set(column_names).issubset(keys | vals):
            raise ColumnsNeedToBeInTableAndVisibleException()
        if Table.ColumnRequirements.IS_KEY in requirements and not set(column_names).issubset(keys):
            raise ColumnsNeedToBeKeysException()
        if Table.ColumnRequirements.IS_VAL in requirements and not set(column_names).issubset(vals):
            raise ColumnsNeedToBeHiddenException()
        if Table.ColumnRequirements.IS_UNIQUE in requirements and not len(set(column_names)) == len(column_names):
            raise ColumnsNeedToBeUniqueException()

    def get_col_with_name(self, name):
        if name in self.keys.keys():
            return self.keys[name]
        if name in self.values.keys():
            return self.values[name]

    def clone(self, column: RawColumn):
        name = self.get_fresh_name(column.name)
        new_node = self.schema.clone(column.node, name)
        return RawColumn(name, new_node, column.keyed_by, column.type, [], self)

    def get_column_from_index(self, index: int):
        if 0 <= index < self.marker:
            return self.keys[index]
        if index < len(self.displayed_columns):
            return self.values[index]

    def get_columns_as_lists(self) -> tuple[list[RawColumn], list[RawColumn], list[RawColumn]]:
        keys = [self.keys[c] for c in self.displayed_columns[:self.marker]]
        vals = [self.values[c] for c in self.displayed_columns[self.marker:]]
        hids = list(set(self.hidden_keys.values()))
        return keys, hids, vals

    def set_keys(self, new_keys: dict[str, RawColumn]):
        self.keys = copy.copy({k: RawColumn.assign_new_table(v, self) for k, v in new_keys.items()})

    def set_vals(self, new_values: dict[str, RawColumn]):
        self.values = copy.copy({k: RawColumn.assign_new_table(v, self) for k, v in new_values.items()})

    def set_hidden_keys(self, new_hidden_keys: dict[str, RawColumn]):
        self.hidden_keys = copy.copy({k: RawColumn.assign_new_table(v, self) for k, v in new_hidden_keys.items()})

    def compose(self, from_keys: list[str], to_key: str, via: list[str] = None):
        self.verify_columns(from_keys, {Table.ColumnRequirements.IS_UNIQUE})
        self.verify_columns([to_key], {Table.ColumnRequirements.IS_KEY})

        key_idx = find_index(self.displayed_columns, to_key)

        # STEP 1
        # Initialise the new table
        # Delete the key we want to compose on from its namespace
        t = Table.create_from_table(self)
        t.namespace -= {to_key}

        # STEP 2
        # 2a. Update the columns to display
        def key_from_name(name: str):
            return t.new_col_from_node(t.schema.get_node_with_name(name), ColumnType.KEY)

        cols_to_add = [key_from_name(key) for key in from_keys]
        t.displayed_columns = (self.displayed_columns[:key_idx] + [str(c) for c in cols_to_add] + self.displayed_columns[key_idx + 1:])
        # 2b. Update the marker
        t.marker = self.marker + len(from_keys) - 1

        # STEP 3
        # Determine the derivation path
        # Want a (backwards) path from from_keys <------- to_key
        # Create start, via, and end nodes
        key = self.keys[to_key]
        start_node = key.node
        via_nodes = None
        if via is not None:
            via_nodes = [t.schema.get_node_with_name(n) for n in via]
        end_node = SchemaNode.product([c.node for c in cols_to_add])
        shortest_p = t.schema.find_shortest_path_between_columns(cols_to_add, key, self.displayed_columns, via_nodes, True)
        derivation, repr, hidden_keys = shortest_p

        # STEP 4
        # Update the keys, values, and hidden keys for the new table
        # If we are replacing u with x, y, z, then for all columns keyed by u, we
        # need to replace u with x, y, z. We also need to update the derivation.
        hidden_columns = [t.new_col_from_node(k, ColumnType.KEY) for k in hidden_keys]

        update_fn = replace_key_and_update_derivation(key, cols_to_add + hidden_columns, derivation)

        t.set_keys(
            {c: update_fn(self.keys[c]) for c in t.displayed_columns[:key_idx]} |
            {t.displayed_columns[i + key_idx]: c for i, c in enumerate(cols_to_add)} |
            {c: update_fn(self.keys[c]) for c in t.displayed_columns[key_idx + len(cols_to_add):t.marker]}
        )

        t.set_vals({c: update_fn(self.values[c]) for c in t.displayed_columns[t.marker:]})

        t.set_hidden_keys(self.hidden_keys | {str(c): c for c in hidden_columns})

        # STEP 5
        # compute the new intermediate representation
        keys, hids, vals = t.get_columns_as_lists()

        old_names = from_keys
        new_names = [str(c) for c in cols_to_add]
        renaming = {old_name: name for old_name, name in zip(old_names, new_names)}

        new_repr = repr[:-1] + [Rename(renaming), repr[-1], End(keys, hids, vals)]
        t.extend_intermediate_representation(new_repr)
        t.execute()

        return t

    # TODO: Handle the case where from columns is zero
    def infer(self, from_columns: list[str], to_column: str, via: list[str] = None, with_name: str = None):
        # An inference from a set of assumption columns to a conclusion column
        self.verify_columns(from_columns, {Table.ColumnRequirements.IS_KEY_OR_VAL, Table.ColumnRequirements.IS_UNIQUE})

        t = Table.create_from_table(self)

        # STEP 1
        # Get name for inferred column
        # Append column to end of displayed columns
        name = str(to_column)
        if with_name is not None:
            name = with_name
        old_name = name
        name = t.get_fresh_name(name)
        t.displayed_columns += [name]

        # STEP 2
        # Get the shortest path in the schema graph
        assumption_columns = [t.get_col_with_name(fc) for fc in from_columns]
        if len(assumption_columns) == 0:
            pass
            # TODO
            # end_node = self.schema.get_node_with_name(to_column)
            # derivation = []
            # repr = [StartTraversal(end_node, Project(end_node), []), EndTraversal(end_node, end_node)]
            # hidden_keys = [end_node]
        else:
            start_node = SchemaNode.product([c.node for c in assumption_columns])
            via_nodes = None
            if via is not None:
                via_nodes = [self.schema.get_node_with_name(n) for n in via]
            end_node = self.schema.get_node_with_name(to_column)
            conclusion_column = RawColumn(name, end_node, [], ColumnType.VALUE, [], t)
            shortest_p = self.schema.find_shortest_path_between_columns(assumption_columns, conclusion_column, t.displayed_columns[:t.marker], via_nodes)
            derivation, repr, hidden_keys = shortest_p

        # 2b. Turn the hidden keys into columns, and rename them if necessary.
        hidden_keys_raw_str = [str(hk) for hk in hidden_keys]  # old names
        hidden_assumptions = [
            t.new_col_from_node(hk, ColumnType.KEY)
            if str(hk) != name
            else RawColumn(name, hk, [], ColumnType.KEY, [], t)
            for hk in hidden_keys]
        hidden_keys_str = [str(c) for c in hidden_assumptions]
        renaming = {old: new for old, new in zip(hidden_keys_raw_str, hidden_keys_str)}

        # STEP 3.
        # Update keys, vals, and hidden keys (keys don't change)
        new_col = RawColumn(name, end_node, assumption_columns + hidden_assumptions, ColumnType.VALUE, derivation, t)

        t.set_vals(t.values | {str(new_col): new_col})
        t.set_hidden_keys(t.hidden_keys | {str(h): h for h in hidden_assumptions})

        keys, hids, vals = t.get_columns_as_lists()

        # STEP 4
        # Update intermediate representation
        new_repr = repr[:-1] + [Rename({old_name: name} | renaming), repr[-1], End(keys, hids, vals)]

        t.extend_intermediate_representation(new_repr)
        t.execute()

        return t

    def combine(self, with_table):
        pass

    def hide(self, column: str):
        self.verify_columns([column], {Table.ColumnRequirements.IS_KEY_OR_VAL})
        t = Table.create_from_table(self)
        idx = find_index(self.displayed_columns, column)
        if idx < self.marker:
            t.marker -= 1
            to_hide = t.keys[column]
        else:
            to_hide = t.values[column]
        t.displayed_columns = t.displayed_columns[:idx] + t.displayed_columns[idx + 1:]
        t.set_hidden_keys(t.hidden_keys | {str(to_hide): to_hide})
        keys, hids, vals = t.get_columns_as_lists()
        t.extend_intermediate_representation([End(keys, hids, vals)])
        t.execute()
        return t

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
        keys = [new_table.keys[c] if c in new_table.keys.keys() else column for c in
                new_table.displayed_columns[:new_table.marker]]
        vals = [new_table.values[c] if c in new_table.values.keys() else column for c in
                new_table.displayed_columns[new_table.marker:]]
        hidden_keys = list(set(self.hidden_keys.values()) - {column})
        new_table.keys = {str(c): RawColumn.assign_new_table(c, new_table) for c in keys}
        new_table.values = {str(c): RawColumn.assign_new_table(c, new_table) for c in vals}
        new_table.hidden_keys = {str(c): RawColumn.assign_new_table(c, new_table) for c in hidden_keys}
        new_table.intermediate_representation = self.intermediate_representation[:-1] + [End(keys, hidden_keys, vals)]
        new_table.df, new_table.dropped_keys_count, new_table.dropped_vals_count, new_table.schema = new_table.schema.execute_query(
            new_table.table_id, self.table_id, new_table.intermediate_representation)
        new_table.schema = self.schema
        return new_table

    def filter(self, predicate: Predicate):
        new_table = Table.create_from_table(self)
        new_table.intermediate_representation = self.intermediate_representation[:-1] + [Filter(predicate),
                                                                                         self.intermediate_representation[
                                                                                             -1]]
        new_table.df, new_table.dropped_keys_count, new_table.dropped_vals_count, new_table.schema = new_table.schema.execute_query(
            new_table.table_id, self.table_id, new_table.intermediate_representation)
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
        was_key = []
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
            elif len(column.keyed_by) == 0:
                was_key += [column]
            else:
                others += [column]
            new_displayed_columns += [col]
        new_values = {}
        new_hidden_keys = {}

        for hk in was_key:
            new_hidden_keys[str(hk)] = RawColumn(hk.name, hk.node, [], ColumnType.VALUE,
                                                 hk.derivation,
                                                 self)
            new_values[str(hk)] = RawColumn(hk.name, hk.node, [hk], ColumnType.VALUE,
                                            hk.derivation,
                                            self)
        for val in others:
            if val not in set(was_key):
                new_values[str(val)] = RawColumn(val.name, val.node, val.keyed_by, ColumnType.VALUE, val.derivation,
                                                 self)

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
                new_values[str(val)] = RawColumn(val.name, val.node, val.keyed_by, ColumnType.VALUE, new_derivation,
                                                 self)

        new_table = Table.create_from_table(self)
        new_table.keys = {k: RawColumn.assign_new_table(v, new_table) for k, v in new_keys.items()}
        new_table.values = {k: RawColumn.assign_new_table(v, new_table) for k, v in new_values.items()}
        new_table.hidden_keys = ({k: RawColumn.assign_new_table(v, new_table) for k, v in self.hidden_keys.items()} |
                                 {k: RawColumn.assign_new_table(v, new_table) for k, v in new_hidden_keys.items()})
        new_table.displayed_columns = new_displayed_columns

        new_table.marker = len(key_list)

        keys = [new_table.keys[c] for c in new_table.displayed_columns[:new_table.marker]]
        vals = [new_table.values[c] for c in new_table.displayed_columns[new_table.marker:]]
        hidden_keys = list(set(new_table.hidden_keys.values()))

        new_table.intermediate_representation = self.intermediate_representation[:-1] + [End(keys, hidden_keys, vals)]
        new_table.df, new_table.dropped_keys_count, new_table.dropped_vals_count, new_table.schema = new_table.schema.execute_query(
            new_table.table_id, self.table_id,
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
            new_table.keys[str(col)] = RawColumn(col.name, col.node, keyed_by, ColumnType.KEY, col.derivation,
                                                 new_table)

        for col in list(self.values.values()):
            if col == column2:
                continue
            keyed_by = [c if c != column2 else column1 for c in col.keyed_by]
            new_table.values[str(col)] = RawColumn(col.name, col.node, keyed_by, ColumnType.VALUE, col.derivation,
                                                   new_table)

        for col in list(self.values.values()):
            if col == column2:
                continue
            keyed_by = [c if c != column2 else column1 for c in col.keyed_by]
            new_table.hidden_keys[str(col)] = RawColumn(col.name, col.node, keyed_by, col.type, col.derivation,
                                                        new_table)

        keys = [new_table.keys[c] for c in new_table.displayed_columns[:new_table.marker]]
        vals = [new_table.values[c] for c in new_table.displayed_columns[new_table.marker:]]
        hidden_keys = list(set(self.hidden_keys.values()))

        new_table.intermediate_representation = self.intermediate_representation[:-1] + [
            Filter(Column(column1) == Column(column2)), End(keys, hidden_keys, vals)]
        new_table.df, new_table.dropped_keys_count, new_table.dropped_vals_count, new_table.schema = new_table.schema.execute_query(
            new_table.table_id, self.table_id,
            new_table.intermediate_representation)
        new_table.schema = self.schema
        return new_table

    def assign(self, name: str, function: Function | AggregationFunction):
        if isinstance(function, Column):
            function = Function.identity(function)

        name = self.get_fresh_name(name)
        if isinstance(function, Function):
            arguments = function.arguments
            function.explicit_keys = self.displayed_columns[:self.marker]
            columns = [a for a in arguments if isinstance(a, Column)]
            columns_dedup = [c.raw_column for c in set(columns)]
            start_columns = [c.raw_column for c in columns]
            nodes = [c.node for c in start_columns]
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
            column = RawColumn(name, end_node, columns_dedup, ColumnType.VALUE, [edge], self)

        elif isinstance(function, AggregationFunction):
            start_column = function.column
            explicit_keys = start_column.get_explicit_keys()
            start_columns = [c for c in explicit_keys] + [start_column.raw_column]
            nodes = [c.node for c in explicit_keys] + [start_column.raw_column.node]
            start_node = SchemaNode.product(nodes)
            new_start_node = SchemaNode.product([c.node for c in explicit_keys])
            end_node = self.schema.add_node(name)
            edge = self.schema.add_edge(start_node, end_node, Cardinality.MANY_TO_ONE)
            self.schema.map_edge_to_closure_function(edge, function)
            self.schema.add_edge(new_start_node, end_node, Cardinality.MANY_TO_ONE)
            column = RawColumn(name, end_node, list(explicit_keys), ColumnType.VALUE, [edge], self)
            hidden_keys = self.hidden_keys
        else:
            raise Exception()

        new_table = Table.create_from_table(self)
        new_table.displayed_columns += [str(column)]
        new_table.values = self.values | {str(column): column}
        keys = [new_table.keys[c] for c in new_table.displayed_columns[:new_table.marker]]
        vals = [new_table.values[c] for c in new_table.displayed_columns[new_table.marker:]]

        new_table.keys = {str(k): RawColumn.assign_new_table(k, new_table) for k in keys}
        new_table.values = {str(v): RawColumn.assign_new_table(v, new_table) for v in vals}

        traversal = [
            StartTraversal(start_columns, Traverse(start_node, end_node), new_table.displayed_columns[:new_table.marker]),
            EndTraversal(start_columns, column)]

        new_table.intermediate_representation = self.intermediate_representation[:-1] + traversal + [
            End(keys, hidden_keys, vals)]
        new_table.df, new_table.dropped_keys_count, new_table.dropped_vals_count, new_table.schema = new_table.schema.execute_query(
            new_table.table_id, self.table_id,
            new_table.intermediate_representation)
        new_table.schema = self.schema
        return new_table

    def sort(self, cols: list[str]):
        new_table = Table.create_from_table(self)
        new_table.intermediate_representation = self.intermediate_representation[:-1] + [Sort(cols),
                                                                                         self.intermediate_representation[
                                                                                             -1]]
        new_table.df, new_table.dropped_keys_count, new_table.dropped_vals_count, new_table.schema = self.schema.execute_query(
            new_table.table_id, self.table_id,
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
