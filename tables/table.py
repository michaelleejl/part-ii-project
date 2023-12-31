import copy
import uuid
from enum import Enum

import numpy as np
import pandas as pd

from helpers.compose_cardinality import compose_cardinality
from helpers.find_cardinality_from_new_key_to_hidden_key import find_cardinality_from_new_key_to_hidden_key
from helpers.find_cardinality_from_new_key_to_new_value import find_cardinality_from_new_key_to_new_value
from helpers.min_cardinality import min_cardinality
from schema import SchemaEdge, Cardinality, reverse_cardinality, BaseType
from schema.helpers.find_index import find_index
from schema.node import SchemaNode, AtomicNode, SchemaClass
from tables.aexp import Aexp
from tables.aggregation import AggregationFunction
from tables.column import Column
from tables.exceptions import KeyMismatchException, ColumnsNeedToBeUniqueException, \
    ColumnsNeedToBeInTableAndVisibleException, ColumnsNeedToBeKeysException, ColumnsNeedToBeInTableException, \
    ColumnsNeedToBeHiddenException
from tables.exp import Exp
from tables.function import Function
from tables.bexp import Bexp
from tables.raw_column import RawColumn, ColumnType
from tables.derivation import DerivationStep, Rename, End, Filter, StartTraversal, EndTraversal, Traverse, Sort, Get


def binary_search(l: list, v):
    low = 0
    high = len(l)
    while low < high:
        mid = (low + high) // 2
        if v < l[mid]:
            high = mid
        elif v > l[mid]:
            low = mid + 1
        else:
            return mid
    return low


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


input_column_type = AtomicNode | Column


def get_name_and_node_from_input_type(input_column: input_column_type) -> tuple[str, AtomicNode]:
    if isinstance(input_column, AtomicNode | SchemaClass):
        return input_column.name, input_column
    if isinstance(input_column, Column):
        return input_column.raw_column.name, input_column.raw_column.node


def get_names_and_nodes_from_input_type(input_columns: list[input_column_type]) -> tuple[list[str], list[AtomicNode]]:
    return tuple([list(x) for x in zip(*[get_name_and_node_from_input_type(ic) for ic in input_columns])])


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
    def construct(cls, columns: list[tuple[AtomicNode, str]], schema):
        table_id = uuid.uuid4().hex
        table = Table(table_id, [], schema)
        keys = []
        for key_node, name in columns:
            key = table.create_column(name, key_node, [], [], True, None, ColumnType.KEY)
            keys += [key]

        table.displayed_columns = [str(k) for k in keys]
        table.keys = {str(k): k for k in keys}
        table.dropped_keys_count = 0
        table.dropped_vals_count = 0
        table.marker = len(keys)
        table.extend_intermediate_representation([Get(keys)])

        table.execute()
        return table

    def extend_intermediate_representation(self, with_new_representation: list[DerivationStep] | None = None):
        if with_new_representation is None:
            with_new_representation = []
        if len(self.intermediate_representation) > 0:
            self.intermediate_representation = self.intermediate_representation[:-1] + with_new_representation
        else:
            self.intermediate_representation = with_new_representation
        keys, hids, vals = self.get_columns_as_lists()
        self.intermediate_representation += [End(keys, hids, vals)]

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

    def create_column(self, name, node: AtomicNode, strong_keys, hidden_keys, is_strong_key_for_self, cardinality,
                      type: ColumnType) -> RawColumn:
        name = self.get_fresh_name(name)
        return RawColumn(name, node, strong_keys, hidden_keys, is_strong_key_for_self, cardinality, type, self)

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

    def new_col_from_node(self, node: AtomicNode, type: ColumnType, name: str = None):
        if name is None:
            name = self.get_fresh_name(node.name)
        else:
            name = self.get_fresh_name(name)
        original_name = node.name
        if name != original_name:
            new_node = self.schema.clone(node, name)
        else:
            new_node = node
        return RawColumn(new_node.name, new_node, [], [], True, None, type, table=self)

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
        return RawColumn(name, new_node, column.strong_keys, column.hidden_keys, column.is_strong_key_for_self,
                         column.cardinality, column.type, self)

    def get_column_from_index(self, index: int):
        if 0 <= index < self.marker:
            return self.keys[self.displayed_columns[index]]
        if index < len(self.displayed_columns):
            return self.values[self.displayed_columns[index]]
    def get_columns_as_lists(self) -> tuple[list[RawColumn], list[RawColumn], list[RawColumn]]:
        keys = [self.keys[c] for c in self.displayed_columns[:self.marker]]
        vals = [self.values[c] for c in self.displayed_columns[self.marker:]]
        hids = list(set(self.hidden_keys.values()))
        return keys, hids, vals

    def set_displayed_columns(self, new_columns):
        self.namespace = set(new_columns)
        self.displayed_columns = new_columns

    def set_keys(self, new_keys: dict[str, RawColumn]):
        self.keys = copy.copy({k: RawColumn.assign_new_table(v, self) for k, v in new_keys.items()})

    def set_vals(self, new_values: dict[str, RawColumn]):
        self.values = copy.copy({k: RawColumn.assign_new_table(v, self) for k, v in new_values.items()})

    def set_hidden_keys(self, new_hidden_keys: dict[str, RawColumn]):
        self.hidden_keys = copy.copy({k: RawColumn.assign_new_table(v, self) for k, v in new_hidden_keys.items()})

    def find_strong_keys_for_column(self, inferred_from: list[RawColumn]):
        strong_keys = set()
        for column in inferred_from:
            strong_keys_of_column = column.get_strong_keys()
            strong_keys = strong_keys.union(strong_keys_of_column)

        return list(sorted(strong_keys, key=lambda x: find_index(x.name, self.displayed_columns)))

    def find_hidden_keys_for_column(self, inferred_from: list[RawColumn]):
        hidden_keys = set()
        for column in inferred_from:
            hidden_keys_of_column = column.get_hidden_keys()
            hidden_keys = hidden_keys.union(hidden_keys_of_column)

        return list(hidden_keys)

    def hide_strong_key(self, key: RawColumn):
        keys, hids, vals = self.get_columns_as_lists()
        for column in keys + hids + vals:
            strong_keys = column.get_strong_keys()
            idx = find_index(key, strong_keys)
            if idx >= 0:
                column.set_strong_keys(strong_keys[:idx] + strong_keys[idx + 1:])
                column.set_hidden_keys(column.get_hidden_keys() + [strong_keys[idx]])

    def show_strong_key(self, key: RawColumn):
        keys, hids, vals = self.get_columns_as_lists()
        for column in keys + hids + vals:
            strong_keys = column.get_strong_keys()
            hidden_keys = column.get_hidden_keys()

            idx = find_index(key, hidden_keys)

            if idx >= 0:
                column.set_hidden_keys(hidden_keys[:idx] + hidden_keys[idx + 1:])
                idxs = [find_index(c.name, self.displayed_columns) for c in strong_keys]
                strong_key_idx = find_index(key.name, self.displayed_columns)
                ins = binary_search(idxs, strong_key_idx)
                column.set_strong_keys(strong_keys[:ins] + [key] + strong_keys[ins + 1:])

    def replace_strong_key(self,
                           to_replace: RawColumn,
                           replace_with_explicit: list[RawColumn],
                           replace_with_hidden: list[RawColumn],
                           cardinality: Cardinality):
        keys, hids, vals = self.get_columns_as_lists()
        for column in keys + hids + vals:
            strong_keys = column.get_strong_keys()
            idx = find_index(to_replace, strong_keys)
            if idx > 0:
                new_strong_keys = set(strong_keys[:idx] + replace_with_explicit + strong_keys[idx + 1:])
                new_strong_keys = list(
                    sorted(new_strong_keys, key=lambda x: find_index(x.name, self.displayed_columns)))
                column.set_strong_keys(new_strong_keys)
                column.set_hidden_keys(list(set(column.get_hidden_keys() + replace_with_hidden)))
                new_cardinality = compose_cardinality(cardinality, column.cardinality)
                column.set_cardinality(new_cardinality)

    def get_representation(self, start: list[RawColumn], end: list[RawColumn], keys, via, backwards):
        shortest_p = self.schema.find_shortest_path_between_columns(start, end, keys, via, backwards)
        cardinality, repr, hidden_keys = shortest_p
        new_repr = []
        hidden_columns = []
        for step in repr:
            from tables.derivation import Traverse, Expand
            if isinstance(step, Traverse) or isinstance(step, Expand):
                step_hidden_keys = step.hidden_keys
                columns = [self.new_col_from_node(hk, ColumnType.KEY) for hk in step_hidden_keys]
                if isinstance(step, Traverse):
                    new_repr += [Traverse(step.start_node, step.end_node, step.hidden_keys, columns)]
                else:
                    new_repr += [Expand(step.start_node, step.end_node, step.indices, step.hidden_keys, columns)]
                hidden_columns += columns
            else:
                new_repr += [step]
        return cardinality, new_repr, hidden_columns

    def find_index_to_insert(self, column, table):
        # must_be_before tracks the LAST place where it may be inserted
        must_be_before = len(self.displayed_columns) - 1
        # the first place
        must_be_after = 0
        for (i, x) in enumerate(self.displayed_columns):
            if i < self.marker:
                c = self.keys[x]
            else:
                c = self.values[x]
            # if the column keys c, then the column must be before c
            if column in set(c.get_strong_keys() + c.get_hidden_keys()):
                must_be_before = min(must_be_before, i)
            if c in set(column.get_strong_keys() + c.get_hidden_keys()):
                must_be_after = max(must_be_after, i)
        assert must_be_after < must_be_before
        # for a valid range, consider the start of that range
        idx = must_be_after
        # do we insert it as a key or a value?
        if idx <= self.marker:
            idx = min(must_be_before, table.marker)
            if column.type == ColumnType.KEY or str(column) in self.displayed_columns:
                table.marker += 1
        return idx

    def compose(self, from_keys: list[input_column_type], to_key: str, via: list[str] = None):
        from_keys_names, from_keys_nodes = get_names_and_nodes_from_input_type(from_keys)
        self.verify_columns(from_keys_names, {Table.ColumnRequirements.IS_UNIQUE})
        self.verify_columns([to_key], {Table.ColumnRequirements.IS_KEY})

        key_idx = find_index(to_key, self.displayed_columns)

        # STEP 1
        # Initialise the new table
        # Delete the key we want to compose on from its namespace
        t = Table.create_from_table(self)
        t.namespace -= {to_key}

        # STEP 2
        # 2a. Update the columns to display
        def key_from_node(node: AtomicNode):
            return t.new_col_from_node(node, ColumnType.KEY)

        cols_to_add = [key_from_node(key) for key in from_keys_nodes]
        t.set_displayed_columns(
                self.displayed_columns[:key_idx] + [str(c) for c in cols_to_add] + self.displayed_columns[
                                                                                   key_idx + 1:])
        # 2b. Update the marker
        t.marker = self.marker + len(from_keys) - 1

        # STEP 3
        # Determine the derivation path
        # Want a (backwards) path from from_keys <------- to_key
        # Create start, via, and end nodes
        key = self.keys[to_key]
        via_nodes = None
        if via is not None:
            via_nodes = [t.schema.get_node_with_name(n) for n in via]
        shortest_p = t.get_representation([key], cols_to_add, t.displayed_columns, via_nodes, True)
        cardinality, repr, hidden_columns = shortest_p

        # STEP 4
        # Update the keys, values, and hidden keys for the new table
        # If we are replacing u with x, y, z, then for all columns keyed by u, we
        # need to replace u with x, y, z. We also need to update the derivation.
        t.set_keys(
            {c: self.keys[c] for c in t.displayed_columns[:key_idx]} |
            {t.displayed_columns[i + key_idx]: c for i, c in enumerate(cols_to_add)} |
            {c: self.keys[c] for c in t.displayed_columns[key_idx + len(cols_to_add):t.marker]}
        )

        t.set_vals({c: self.values[c] for c in t.displayed_columns[t.marker:]})

        t.set_hidden_keys(self.hidden_keys | {str(c): c for c in hidden_columns})

        t.replace_strong_key(key, cols_to_add, hidden_columns, cardinality)

        # STEP 5
        # compute the new intermediate representation
        t.extend_intermediate_representation(repr)
        t.execute()

        return t

    def deduce(self, function: Exp, with_name: str):
        t = Table.create_from_table(self)
        name = t.get_fresh_name(with_name)
        if isinstance(function, Exp):
            exp, start_columns = Exp.convert_exp(function)
            nodes = [c.node for c in start_columns]
            start_node = SchemaNode.product(nodes)
            strong_keys = t.find_strong_keys_for_column(start_columns)
            hidden_keys = [c.get_hidden_keys() for c in start_columns]
            acc = set()
            for hk in hidden_keys:
                if acc.issubset(hk) or hk.issubset(acc):
                    acc |= set(hk)
                else:
                    raise KeyMismatchException(acc, hk)
            hidden_keys = list(acc)
            if isinstance(exp, Aexp):
                node_type = BaseType.FLOAT
            elif isinstance(exp, Bexp):
                node_type = BaseType.name
            else:
                raise NotImplemented()
            end_node = self.schema.add_node(AtomicNode(name, node_type))
            # TODO: Consider special ONE-TO-ONE Functions
            edge = self.schema.add_edge(start_node, end_node, Cardinality.MANY_TO_ONE)
            self.schema.map_edge_to_closure_function(edge, exp, len(start_columns))
            cardinality = Cardinality.MANY_TO_ONE
            for column in start_columns:
                cardinality = compose_cardinality(column.cardinality, cardinality)
            column = RawColumn(name, end_node, strong_keys, hidden_keys, False, cardinality, ColumnType.VALUE, t)

        # elif isinstance(function, AggregationFunction):
        #     start_column = function.column
        #     explicit_keys = start_column.get_strong_keys()
        #     start_columns = [c for c in explicit_keys] + [start_column.raw_column]
        #     nodes = [c.node for c in explicit_keys] + [start_column.raw_column.node]
        #     start_node = SchemaNode.product(nodes)
        #     new_start_node = SchemaNode.product([c.node for c in explicit_keys])
        #     end_node = self.schema.add_node(name)
        #     edge = self.schema.add_edge(start_node, end_node, Cardinality.MANY_TO_ONE)
        #     self.schema.map_edge_to_closure_function(edge, function)
        #     self.schema.add_edge(new_start_node, end_node, Cardinality.MANY_TO_ONE)
        #     column = RawColumn(name, end_node, list(explicit_keys), ColumnType.VALUE, self)
        else:
            raise Exception()

        t.set_displayed_columns(t.displayed_columns + [str(column)])
        t.set_vals(t.values | {str(column): column})
        traversal = [
            StartTraversal(start_columns, t.displayed_columns[:t.marker]),
            Traverse(start_node, end_node),
            EndTraversal(start_columns, [column])]

        t.extend_intermediate_representation(traversal)
        t.execute()
        return t

    # TODO: Handle the case where from columns is zero
    def infer(self, from_columns: list[str], to_column: input_column_type, via: list[SchemaNode] = None,
              with_name: str = None):
        # An inference from a set of assumption columns to a conclusion column
        self.verify_columns(from_columns, {Table.ColumnRequirements.IS_KEY_OR_VAL, Table.ColumnRequirements.IS_UNIQUE})

        to_column_name, to_column_node = get_name_and_node_from_input_type(to_column)
        if with_name is not None:
            to_column_name = with_name

        t = Table.create_from_table(self)

        # STEP 1
        # Get name for inferred column
        # Append column to end of displayed columns
        old_name = to_column_name
        name = t.get_fresh_name(to_column_name)
        t.set_displayed_columns(t.displayed_columns + [name])

        # STEP 2
        # Get the shortest path in the schema graph
        assumption_columns = [t.get_col_with_name(fc) for fc in from_columns]
        strong_keys = t.find_strong_keys_for_column(assumption_columns)
        old_hidden_keys = t.find_hidden_keys_for_column(assumption_columns)
        cardinalities = [column.cardinality for column in assumption_columns]

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
                via_nodes = via
            conclusion_column = RawColumn(name, to_column_node, [], [], False,
                                          Cardinality.MANY_TO_MANY, ColumnType.VALUE, t)
            shortest_p = t.get_representation(assumption_columns, [conclusion_column],
                                                 t.displayed_columns[:t.marker], via_nodes, False)
            cardinality, repr, hidden_keys = shortest_p

        # 2b. Turn the hidden keys into columns, and rename them if necessary.
        hidden_keys_raw_str = [str(hk) for hk in hidden_keys]  # old names
        hidden_assumptions = [
            hk
            if str(hk) != name
            else RawColumn(name, hk, [], [], True,
                           None, ColumnType.KEY, t)
            for hk in hidden_keys]
        hidden_keys_str = [str(c) for c in hidden_assumptions]
        renaming = {old: new for old, new in zip(hidden_keys_raw_str, hidden_keys_str)}

        # STEP 3.
        # Update keys, vals, and hidden keys (keys don't change)
        for c in cardinalities:
            if c is not None:
                cardinality = compose_cardinality(c, cardinality)

        new_col = RawColumn(name, to_column_node, strong_keys, old_hidden_keys + hidden_assumptions, False, cardinality,
                            ColumnType.VALUE, t)

        t.set_vals(t.values | {str(new_col): new_col})
        t.set_hidden_keys(t.hidden_keys | {str(h): h for h in hidden_assumptions})

        # STEP 4
        # Update intermediate representation

        t.extend_intermediate_representation(repr)
        t.execute()

        return t

    def combine(self, with_table):
        pass

    def hide(self, column: str):
        self.verify_columns([column], {Table.ColumnRequirements.IS_KEY_OR_VAL})
        t = Table.create_from_table(self)
        idx = find_index(column, self.displayed_columns)
        if idx < self.marker:
            t.marker -= 1
            to_hide = t.keys[column]
        else:
            to_hide = t.values[column]
        t.set_displayed_columns( t.displayed_columns[:idx] + t.displayed_columns[idx + 1:])
        t.hide_strong_key(to_hide)
        t.set_hidden_keys(t.hidden_keys | {str(to_hide): to_hide})
        t.extend_intermediate_representation()
        t.execute()
        return t

    def show(self, column: str):
        name = column
        self.verify_columns([name], {Table.ColumnRequirements.IS_HIDDEN})
        t = Table.create_from_table(self)
        column = self.hidden_keys[name]
        idx = self.find_index_to_insert(column, t)
        t.set_displayed_columns(t.displayed_columns[:idx] + [name] + t.displayed_columns[idx:])
        keys = [t.keys[c] if c in t.keys.keys() else column for c in
                t.displayed_columns[:t.marker]]
        vals = [t.values[c] if c in t.values.keys() else column for c in
                t.displayed_columns[t.marker:]]
        hidden_keys = list(set(self.hidden_keys.values()) - {column})
        t.set_keys({str(c): c for c in keys})
        t.set_vals({str(c): c for c in vals})
        t.set_hidden_keys({str(c): c for c in hidden_keys})
        t.show_strong_key(column)
        t.extend_intermediate_representation()
        t.execute()
        return t

    def filter(self, by: str | Bexp):
        t = Table.create_from_table(self)
        if isinstance(by, str):
            by = t.__getitem__(by).to_bexp()
        exp, arguments = Exp.convert_exp(by)
        t.extend_intermediate_representation([Filter(exp, arguments)])
        t.execute()
        return t

    def generate_new_key(self, col: RawColumn):
        # Find the strong keys of the column
        strong_keys = col.get_strong_keys()

        idx = find_index(col.name, self.displayed_columns)

        # Call the strong key s, and the new column k.
        # Either s -> k (remains strong key)
        remains_strong_key = np.array([find_index(k.name, self.displayed_columns) <= idx for k in strong_keys])
        # Or k -> s (becomes value).
        becomes_value = ~remains_strong_key

        new_strong_keys_for_col = np.array(strong_keys)[remains_strong_key]
        new_values_for_col = np.array(strong_keys)[becomes_value]

        # If a column is keyed by a hidden key, h, then it is necessarily the case that
        # k -> h
        hidden_keys = col.get_hidden_keys()

        # Should k become a key for itself?
        # If it was already a key for itself, then it should remain so.
        # If not, and some of its strong keys have become values, then it should become a key for itself.
        # Finally, if there were hidden keys.
        should_be_key_for_self = (col.is_strong_key_for_self or
                                  len(new_strong_keys_for_col) != len(strong_keys)
                                  or len(hidden_keys) > 0)

        new_cardinality = Cardinality.MANY_TO_MANY
        if len(new_strong_keys_for_col) == 0:
            new_cardinality = None
        elif len(new_strong_keys_for_col) == len(strong_keys):
            new_cardinality = col.cardinality

        new_key = RawColumn(col.name, col.node, list(new_strong_keys_for_col), [], should_be_key_for_self,
                            new_cardinality, ColumnType.KEY, self)
        return new_key, new_key.get_strong_keys(), hidden_keys, new_values_for_col

    # TODO: Cleanup
    def set_key(self, new_keys: list[str]):
        self.verify_columns(new_keys, {Table.ColumnRequirements.IS_KEY_OR_VAL})

        t = Table.create_from_table(self)
        new_values = [self.get_column_from_index(i)
                      for (i, c) in enumerate(self.displayed_columns)
                      if c not in new_keys]
        new_keys = [self.get_column_from_index(i)
                    for (i, c) in enumerate(self.displayed_columns)
                    if c in new_keys]
        t.set_displayed_columns( [k.name for k in new_keys + new_values])
        t.marker = len(new_keys)

        keys = {}
        strong_keys_for_new_values = {v: set() for v in new_values}
        cardinalities_for_new_values = {v: Cardinality.MANY_TO_MANY for v in new_values}
        strong_keys_for_new_hidden_keys = {}
        cardinalities_for_new_hidden_keys = {}

        for idx, col in enumerate(new_keys):
            key, new_strong_keys, hidden_keys, new_values_for_key = t.generate_new_key(col)
            keys[key.name] = key

            for value in new_values_for_key:
                strong_keys_for_new_values[value] |= set(new_strong_keys)
                key_to_value_cardinality = find_cardinality_from_new_key_to_new_value(col)
                cardinalities_for_new_values[value] = min_cardinality(key_to_value_cardinality,
                                                                    cardinalities_for_new_values[value])

            for hidden_key in hidden_keys:
                if hidden_key not in strong_keys_for_new_hidden_keys:
                    strong_keys_for_new_hidden_keys[hidden_key] = set()
                    cardinalities_for_new_hidden_keys[hidden_key] = Cardinality.MANY_TO_MANY
                strong_keys_for_new_hidden_keys[hidden_key] |= set(new_strong_keys)
                key_to_hidden_key_cardinality = (find_cardinality_from_new_key_to_hidden_key(key)
                                                 if key.name != hidden_key.name else Cardinality.ONE_TO_ONE)
                cardinalities_for_new_hidden_keys[hidden_key] = min_cardinality(key_to_hidden_key_cardinality,
                                                                                cardinalities_for_new_hidden_keys[
                                                                                    hidden_key])

        values = {}
        hidden_keys = {}

        for col in new_values:
            new_cardinality = cardinalities_for_new_values[col]
            if col.cardinality is not None:
                new_cardinality = min_cardinality(col.cardinality, new_cardinality)

            old_strong_key_names = [c.name for c in col.get_strong_keys()]
            new_strong_keys = strong_keys_for_new_values[col]
            for name in old_strong_key_names:
                if name == col.name:
                    continue
                elif name in keys:
                    old_strong_key = keys[name]
                else:
                    old_strong_key = values[name]
                new_strong_keys |= set(old_strong_key.get_strong_keys())
                if old_strong_key.cardinality is not None:
                    new_cardinality = compose_cardinality(old_strong_key.cardinality, new_cardinality)

            old_hidden_keys = col.get_hidden_keys()
            new_hidden_keys_for_column = []
            for old_hidden_key in old_hidden_keys:
                if old_hidden_key in strong_keys_for_new_hidden_keys:
                    new_strong_keys |= strong_keys_for_new_hidden_keys[old_hidden_key]
                    c1 = cardinalities_for_new_hidden_keys[old_hidden_key]
                    if old_hidden_key.name == col.name:
                        c2 = Cardinality.ONE_TO_ONE
                    elif len(old_hidden_keys) == 1:
                        c2 = Cardinality.MANY_TO_ONE
                    else:
                        c2 = Cardinality.MANY_TO_MANY
                    new_cardinality = min_cardinality(new_cardinality, compose_cardinality(c1, c2))
                else:
                    new_hidden_keys_for_column += [old_hidden_key]
                    hidden_keys[str(old_hidden_key)] = old_hidden_key

            if new_cardinality == Cardinality.ONE_TO_MANY or new_cardinality == Cardinality.MANY_TO_MANY and len(
                    new_hidden_keys_for_column) == 0:
                new_hidden_key = RawColumn(col.name, col.node, [], [], True, None, ColumnType.KEY, t)
                new_hidden_keys_for_column = [new_hidden_key]
                hidden_keys[str(new_hidden_key)] = new_hidden_key
            new_strong_keys = list(sorted(new_strong_keys, key=lambda x: find_index(x.name, t.displayed_columns)))
            values[col.name] = RawColumn(col.name, col.node, new_strong_keys, new_hidden_keys_for_column, False, new_cardinality,
                                  ColumnType.VALUE, t)

        t.set_keys(keys)
        t.set_vals(values)
        t.set_hidden_keys(hidden_keys)

        t.extend_intermediate_representation()
        t.execute()
        return t

    def equate(self, col1, col2):
        self.verify_columns([col1, col2], {Table.ColumnRequirements.IS_KEY_OR_VAL})

        t = Table.create_from_table(self)
        idx = self.displayed_columns.index(col2)
        t.set_displayed_columns([c for c in self.displayed_columns if c != col2])

        if idx < self.marker:
            t.marker -= 1
            column2 = self.keys[col2]
        else:
            column2 = self.values[col2]

        if self.displayed_columns.index(col1) < self.marker:
            column1 = self.keys[col1]
        else:
            column1 = self.values[col1]

        self.replace_strong_key(column2, [column1], [], Cardinality.ONE_TO_ONE)
        rexp = Column(column1) == Column(column2)
        exp, arguments = Exp.convert_exp(rexp)
        t.extend_intermediate_representation([Filter(exp, arguments)])
        t.execute()
        return t

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
            column = RawColumn(name, end_node, columns_dedup, ColumnType.VALUE, self)

        elif isinstance(function, AggregationFunction):
            start_column = function.column
            explicit_keys = start_column.get_strong_keys()
            start_columns = [c for c in explicit_keys] + [start_column.raw_column]
            nodes = [c.node for c in explicit_keys] + [start_column.raw_column.node]
            start_node = SchemaNode.product(nodes)
            new_start_node = SchemaNode.product([c.node for c in explicit_keys])
            end_node = self.schema.add_node(name)
            edge = self.schema.add_edge(start_node, end_node, Cardinality.MANY_TO_ONE)
            self.schema.map_edge_to_closure_function(edge, function)
            self.schema.add_edge(new_start_node, end_node, Cardinality.MANY_TO_ONE)
            column = RawColumn(name, end_node, list(explicit_keys), ColumnType.VALUE, self)
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
            StartTraversal(start_columns, new_table.displayed_columns[:new_table.marker]),
            Traverse(start_node, end_node),
            EndTraversal(start_columns, [column])]

        new_table.intermediate_representation = self.intermediate_representation[:-1] + traversal + [
            End(keys, hidden_keys, vals)]
        new_table.df, new_table.dropped_keys_count, new_table.dropped_vals_count, new_table.schema = new_table.schema.execute_query(
            new_table.table_id, self.table_id,
            new_table.intermediate_representation)
        new_table.schema = self.schema
        return new_table

    def sort(self, cols: list[str]):
        t = Table.create_from_table(self)
        t.extend_intermediate_representation([Sort(cols)])
        t.execute()
        return t

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
