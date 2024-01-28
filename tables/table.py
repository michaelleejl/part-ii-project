import copy
import uuid
from enum import Enum
from typing import Any

import numpy as np
import pandas as pd

from helpers.compose_cardinality import compose_cardinality
from schema import Cardinality, BaseType, is_sublist
from schema.helpers.find_index import find_index
from schema.node import SchemaNode, AtomicNode, SchemaClass
from tables.column import Column
from tables.derivation.derivation_node import DerivationNode, ColumnNode, IntermediateNode, RootNode
from tables.domain import Domain
from tables.exceptions import ColumnsNeedToBeUniqueException, \
    ColumnsNeedToBeInTableAndVisibleException, ColumnsNeedToBeKeysException, ColumnsNeedToBeInTableException, \
    ColumnsNeedToBeValuesException
from tables.exp import Exp, ExtendExp, MaskExp
from tables.helpers.carry_keys_through_path import carry_keys_through_representation
from tables.helpers.transform_step import transform_step
from tables.helpers.wrap_aexp import wrap_aexp
from tables.helpers.wrap_bexp import wrap_bexp
from tables.helpers.wrap_sexp import wrap_sexp
from tables.internal_representation import RepresentationStep, End, Filter, Sort, Get, EndTraversal, StartTraversal, Pop

existing_column = str | Column | ColumnNode
new_column = AtomicNode | SchemaClass | Column | ColumnNode


def flatten(xss):
    return [x for xs in xss for x in xs]


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


def check_for_uniqueness(column_names):
    if len(column_names) != len(set(column_names)):
        raise ColumnsNeedToBeUniqueException()


def classify_groups(children, to_invert):
    groups = {n: [] for n in range(5)}
    for i, child in enumerate(children):
        domains = child.domains
        if len(domains) == 0:
            groups[0] += [child]
            continue
        elif domains == to_invert:
            continue
        elif is_sublist(domains, to_invert):
            groups[1] += [child]
            continue
        elif is_sublist(to_invert, domains):
            groups[2] += [child]
            continue
        elif len(set(to_invert).intersection(set(domains))) > 0:
            groups[3] += [child]
            continue
        elif len(set(to_invert).intersection(set(domains))) == 0:
            groups[4] += [child]
            continue
        else:
            assert False
    return groups


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

    def __init__(self, table_id, derivation: RootNode | None,
                 intermediate_representation: list[RepresentationStep], schema, derived_from=None):
        self.table_id = table_id
        self.derived_from = derived_from
        self.displayed_columns = []
        self.marker = 0  # index of the first value column
        self.intermediate_representation = intermediate_representation
        self.derivation = derivation
        self.schema = schema
        self.df = pd.DataFrame()
        self.dropped_keys_count = 0
        self.dropped_vals_count = 0

    @classmethod
    def construct(cls, columns: list[tuple[AtomicNode, str]], schema):
        table_id = uuid.uuid4().hex
        table = Table(table_id, None, [], schema)
        keys = []
        namespace = set()
        for key_node, name in columns:
            key = table.create_column(name, key_node, namespace)
            keys += [key]
            namespace.add(name)

        table.displayed_columns = [str(k) for k in keys]
        table.left = {str(k): k for k in keys}
        table.dropped_keys_count = 0
        table.dropped_vals_count = 0
        table.marker = len(keys)
        table.derivation = DerivationNode.create_root(keys)
        table.extend_intermediate_representation()

        table.execute()
        return table

    def get_name_and_node(self, input: new_column) -> tuple[str, AtomicNode]:
        if isinstance(input, AtomicNode | SchemaClass):
            return input.name, input
        elif isinstance(input, Column):
            return input.name, input.get_schema_node()
        elif isinstance(input, ColumnNode):
            return input.get_name(), input.get_schema_node()
        # elif isinstance(input, str):
        #     node = self.derivation.find_column_with_name(input)
        #     return node.get_name(), node.get_schema_node()

    def get_names_and_nodes(self, input_columns: list[new_column]) -> tuple[list[Any], ...]:
        return tuple([list(x) for x in zip(*[self.get_name_and_node(ic) for ic in input_columns])])

    def get_existing_column(self, input: existing_column) -> ColumnNode:
        if isinstance(input, str):
            return self.derivation.find_column_with_name(input)
        elif isinstance(input, Column):
            return input.node
        elif isinstance(input, ColumnNode):
            return input

    def get_existing_columns(self, inputs: list[existing_column]) -> list[ColumnNode]:
        columns = []
        for input in inputs:
            columns += [self.get_existing_column(input)]
        return columns

    def extend_intermediate_representation(self, with_new_representation: list[RepresentationStep] | None = None):
        print("DERIVATION")
        print(self.derivation)
        if with_new_representation is None:
            with_new_representation = []
        self.intermediate_representation = self.derivation.to_intermediate_representation()
        if len(self.intermediate_representation) > 0:
            self.intermediate_representation = self.intermediate_representation + with_new_representation
        else:
            self.intermediate_representation = with_new_representation
        left, hids, right = self.get_columns_as_lists()
        print(self.derivation)
        self.intermediate_representation += [End(left, hids, right)]

    def execute(self):
        self.df, self.dropped_keys_count, self.dropped_vals_count, self.schema = self.schema.execute_query(
            self.table_id,
            self.derived_from,
            self.intermediate_representation)

    @classmethod
    def create_from_table(cls, table):
        table_id = uuid.uuid4().hex
        new_table = Table(table_id,
                          table.derivation,
                          table.intermediate_representation,
                          table.schema,
                          table.table_id)
        new_table.displayed_columns = copy.copy(table.displayed_columns)
        new_table.marker = table.marker
        new_table.dropped_keys_count = table.dropped_keys_count
        new_table.dropped_vals_count = table.dropped_vals_count
        return new_table

    def create_column(self, name, node: AtomicNode, namespace) -> Domain:
        name = self.get_fresh_name(name, namespace)
        return Domain(name, node)

    def unpack_inputs(self):
        pass

    def get_fresh_name(self, name: str, namespace: set[str]):
        candidate = name
        if candidate in namespace:
            i = 1
            candidate = f"{name}_{i}"
            while candidate in namespace:
                i += 1
                candidate = f"{name}_{i}"
        return candidate

    def new_col_from_node(self, namespace: set[str], node: AtomicNode, name: str = None):
        if name is None:
            name = self.get_fresh_name(node.name, namespace)
        else:
            name = self.get_fresh_name(name, namespace)
        return Domain(name, node)

    def verify_columns(self, columns: list[ColumnNode], requirements: set[ColumnRequirements]):
        keys = set(self.derivation.get_keys())
        vals = set(self.derivation.get_values())
        hids = set(self.derivation.get_hidden())
        if Table.ColumnRequirements.IS_UNIQUE in requirements and len(set(columns)) != len(columns):
            raise ColumnsNeedToBeUniqueException()
        if Table.ColumnRequirements.IS_KEY_OR_VAL_OR_HIDDEN in requirements and not set(columns).issubset(
                keys | vals | hids):
            raise ColumnsNeedToBeInTableException()
        if Table.ColumnRequirements.IS_KEY_OR_VAL in requirements and not set(columns).issubset(keys | vals):
            raise ColumnsNeedToBeInTableAndVisibleException()
        if Table.ColumnRequirements.IS_KEY in requirements and not set(columns).issubset(keys):
            raise ColumnsNeedToBeKeysException()
        if Table.ColumnRequirements.IS_VAL in requirements and not set(columns).issubset(vals):
            raise ColumnsNeedToBeValuesException()
        if Table.ColumnRequirements.IS_UNIQUE in requirements and not len(set(columns)) == len(columns):
            raise ColumnsNeedToBeUniqueException()

    def get_col_with_name(self, name) -> Domain | None:
        return self.derivation.find_column_with_name(name)

    def get_column_from_index(self, index: int) -> ColumnNode | None:
        if index < len(self.displayed_columns):
            return self.derivation.find_column_with_name(self.displayed_columns[index])

    def get_columns_as_lists(self) -> tuple[list[ColumnNode], list[Domain], list[ColumnNode]]:
        keys_and_values = self.derivation.get_keys_and_values()
        visible = list(sorted(keys_and_values, key=lambda c: find_index(c.name, self.displayed_columns)))
        left = visible[:self.marker]
        right = visible[self.marker:]
        hids = self.derivation.get_hidden()
        return left, hids, right

    def set_displayed_columns(self, new_columns):
        self.displayed_columns = new_columns

    def find_strong_keys_for_column(self, inferred_from: list[ColumnNode]):
        strong_keys = set()
        for column in inferred_from:
            strong_keys_of_column = column.get_strong_keys()
            if len(strong_keys_of_column) == 0:
                strong_keys = strong_keys.union(column.domains)
            else:
                strong_keys = strong_keys.union(strong_keys_of_column)

        return list(sorted(strong_keys, key=lambda x: find_index(x.name, self.displayed_columns)))

    def find_hidden_keys_for_column(self, inferred_from: list[ColumnNode], aggregated_over: set[ColumnNode]):
        hidden_keys = set()
        blocked_hidden_keys = self.blocked_hidden_keys()
        for column in inferred_from:
            hidden_keys_of_column = column.get_hidden_keys()
            if column.get_domain() in aggregated_over:
                hidden_keys = hidden_keys.union(set(hidden_keys_of_column).intersection(blocked_hidden_keys))
            else:
                hidden_keys = hidden_keys.union(hidden_keys_of_column)
        return list(sorted(hidden_keys, key=lambda x: find_index(x, self.derivation.get_hidden())))

    def get_namespace(self):
        return set([d.name for d in self.derivation.get_keys_and_values() + self.derivation.get_hidden()])

    def get_representation(self, start: list[Domain], end: list[Domain], via, backwards, aggregated_over, namespace):
        shortest_p = self.schema.find_shortest_path_between_columns(start, end, via, backwards)
        cardinality, repr, hidden_keys = shortest_p
        new_repr: list[RepresentationStep] = []
        hidden_columns = []
        get_next_step = transform_step(namespace, self, start, end, aggregated_over)
        for step in repr:
            next_step, cols = get_next_step(step)
            new_repr += [next_step]
            hidden_columns += cols
        return cardinality, new_repr, hidden_columns

    def find_index_to_insert(self, column, table):
        # must_be_before tracks the LAST place where it may be inserted
        must_be_before = len(self.displayed_columns) - 1
        # the first place
        must_be_after = 0
        for (i, x) in enumerate(self.displayed_columns):
            c = self.derivation.find_column_with_name(x)
            # if the column keys c, then the column must be before c
            if column in set(c.get_strong_keys() + c.get_hidden_keys()):
                must_be_before = min(must_be_before, i)
            if c in set(column.get_strong_keys() + c.get_hidden_keys()):
                must_be_after = max(must_be_after, i)
        assert must_be_after < must_be_before
        # for a valid range, consider the start of that range
        idx = must_be_after
        # do we insert it as a key or a value? default to key
        if idx <= self.marker:
            idx = min(must_be_before, table.marker)
            table.marker += 1
        return idx

    def compose(self, from_keys: list[new_column], to_key: existing_column, via: list[str] = None):
        from_keys_names, from_keys_nodes = self.get_names_and_nodes(from_keys)
        self.verify_columns(from_keys_names, {Table.ColumnRequirements.IS_UNIQUE})
        key = self.get_existing_column(to_key)
        self.verify_columns([key], {Table.ColumnRequirements.IS_KEY})

        key_idx = find_index(to_key, self.displayed_columns)

        # STEP 1
        # Initialise the new table
        # Delete the key we want to compose on from its namespace
        t = Table.create_from_table(self)

        namespace = t.get_namespace()
        namespace.remove(to_key)
        # STEP 2
        # 2a. Update the columns to display

        cols_to_add = []
        for k in from_keys_nodes:
            col = t.new_col_from_node(namespace, k)
            namespace.add(col.name)
            cols_to_add += [col]

        t.set_displayed_columns(
            self.displayed_columns[:key_idx] + [str(c) for c in cols_to_add] + self.displayed_columns[
                                                                               key_idx + 1:])
        # 2b. Update the marker
        t.marker = self.marker + len(from_keys) - 1

        # STEP 3
        # Determine the derivation path
        # Want a (backwards) path from from_keys <------- to_key
        # Create start, via, and end nodes
        via_nodes = None
        if via is not None:
            via_nodes = [t.schema.get_node_with_name(n) for n in via]
        shortest_p = t.get_representation(cols_to_add, [key.get_domain()], via_nodes, False, [], namespace)
        cardinality, repr, hidden_columns = shortest_p

        # STEP 4
        # compute the new intermediate representation
        t.derivation = self.derivation.compose(cols_to_add, key.get_domain(), hidden_columns, repr, cardinality)
        t.extend_intermediate_representation()
        t.execute()

        return t

    def blocked_hidden_keys(self):
        left, _, _ = self.get_columns_as_lists()
        hidden = set(flatten([c.get_hidden_keys() for c in left]))
        return hidden

    def deduce(self, function: Exp, with_name: str):
        t = Table.create_from_table(self)
        exp, start_columns, aggregated_over = Exp.convert_exp(function)
        nodes = [c.node for c in start_columns]
        start_columns = [c for c in start_columns]
        start_node = SchemaNode.product(nodes)
        node_type = function.exp_type
        end_node = self.schema.add_node(AtomicNode(with_name, node_type))
        # TODO: Consider special ONE-TO-ONE Functions
        edge = self.schema.add_edge(start_node, end_node, Cardinality.MANY_TO_ONE)
        modified_start_node = None
        modified_start_cols = None
        if len(aggregated_over) > 0:
            modified_start_cols = [i for i, c in enumerate(start_columns) if c not in aggregated_over]
            modified_start_node = SchemaNode.product([start_columns[i].node for i in modified_start_cols])
        self.schema.map_edge_to_closure_function(edge, exp, len(start_columns), modified_start_node,
                                                 modified_start_cols)
        return t.infer_internal([col.name for col in start_columns], end_node, with_name=with_name,
                                aggregated_over=aggregated_over)

    def extend(self, column: existing_column, with_function, with_name: str):
        column = self.get_existing_column(column)
        self.verify_columns([column], {self.ColumnRequirements.IS_VAL})
        value: ColumnNode = column
        strong_keys = value.get_strong_keys()
        hidden_keys = [hk for hk in value.get_hidden_keys() if hk.name != column]

        match value.get_schema_node().node_type:
            case BaseType.FLOAT:
                with_function = wrap_aexp(with_function)
            case BaseType.BOOL:
                with_function = wrap_bexp(with_function)
            case BaseType.STRING:
                with_function = wrap_sexp(with_function)

        assert len(strong_keys + hidden_keys) > 0
        assert value.get_schema_node().node_type == with_function.exp_type
        function = ExtendExp([], value.domains[0], with_function, with_function.exp_type)
        return self.deduce(function, with_name)

    def mask(self, column: existing_column, mask_with, with_name: str):
        column = self.get_existing_column(column)
        self.verify_columns([column], {self.ColumnRequirements.IS_KEY_OR_VAL})
        strong_keys = self.find_strong_keys_for_column([column])
        hidden_keys = [hk for hk in column.get_hidden_keys() if hk.name != column]

        mask_with = wrap_bexp(mask_with)
        assert len(strong_keys + hidden_keys) > 0
        function = MaskExp([], column.domains[0], mask_with, column.get_schema_node().node_type)
        return self.deduce(function, with_name)

    def infer(self, from_columns: list[existing_column], to_column: new_column, via: list[SchemaNode] = None,
              with_name: str = None):
        return self.infer_internal(from_columns, to_column, via, with_name)

    def infer_internal(self, from_columns: list[existing_column], to_column: new_column, via: list[SchemaNode] = None,
                       with_name: str = None, aggregated_over: list[Domain] = None):
        # An inference from a set of assumption columns to a conclusion column
        assumption_columns = self.get_existing_columns(from_columns)
        self.verify_columns(assumption_columns,
                            {Table.ColumnRequirements.IS_KEY_OR_VAL, Table.ColumnRequirements.IS_UNIQUE})

        to_column_name, to_column_node = self.get_name_and_node(to_column)
        if with_name is not None:
            to_column_name = with_name



        t = Table.create_from_table(self)

        namespace = t.get_namespace()

        # STEP 1
        # Get name for inferred column
        # Append column to end of displayed columns
        name = t.get_fresh_name(to_column_name, namespace)
        t.set_displayed_columns(t.displayed_columns + [name])

        namespace |= {name}

        strong_keys = t.find_strong_keys_for_column(assumption_columns)

        if isinstance(to_column, Column) or isinstance(to_column, ColumnNode):
            if isinstance(to_column, Column):
                col = to_column.node
            else:
                col = to_column
            root = col.find_root_of_tree()
            key_node = root.find_node_with_domains(strong_keys)
            path = key_node.path_to_value(col)

            #todo: cleanup hidden keys etc
            new_root = t.derivation.insert_key(strong_keys)
            parent = key_node
            for child in path[1:]:
                new_root = new_root.add_child(parent, child)
                parent = child

            t.derivation = new_root
            t.extend_intermediate_representation()
            t.execute()
            return t

            # STEP 2
        # Get the shortest path in the schema graph
        if aggregated_over is None:
            aggregated_over = []
        old_hidden_keys = t.find_hidden_keys_for_column(assumption_columns, set(aggregated_over))
        cardinalities = [column.cardinality for column in assumption_columns]
        conclusion_column = Domain(name, to_column_node)

        if len(assumption_columns) == 0:
            pass
            strong_keys = []
            hidden_key = Domain(t.get_fresh_name(name, namespace), to_column_node)
            repr = [Pop(), Get([hidden_key]), StartTraversal([hidden_key]),
                    EndTraversal([hidden_key], [conclusion_column])]
            hidden_keys = [hidden_key]
            cardinality = Cardinality.MANY_TO_MANY
        else:
            via_nodes = None
            if via is not None:
                via_nodes = via

            shortest_p = t.get_representation([c.get_domain() for c in assumption_columns], [conclusion_column],
                                              via_nodes, False, aggregated_over, namespace)
            cardinality, repr, hidden_keys = shortest_p

        # 2b. Turn the hidden keys into columns, and rename them if necessary.
        hidden_assumptions = [
            hk
            if str(hk) != name
            else hk
            for hk in hidden_keys]

        # STEP 3.
        # Update keys, vals, and hidden keys (keys don't change)
        for c in cardinalities:
            if c is not None:
                cardinality = compose_cardinality(c, cardinality)

        new_col = Domain(name, to_column_node)

        # TODO: Prepend
        # STEP 4
        # Update intermediate representation
        t.derivation = self.derivation.infer(new_col, strong_keys, old_hidden_keys, hidden_assumptions,
                                             [c for c in assumption_columns], cardinality, repr)
        t.extend_intermediate_representation()
        t.execute()

        return t

    def hide(self, column: existing_column):
        column = self.get_existing_column(column)
        self.verify_columns([column], {Table.ColumnRequirements.IS_KEY})
        t = Table.create_from_table(self)
        idx = find_index(column.name, self.displayed_columns)
        if idx < t.marker:
            t.marker -= 1
        t.set_displayed_columns(t.displayed_columns[:idx] + t.displayed_columns[idx + 1:])
        t.derivation = self.derivation.hide(column)
        t.extend_intermediate_representation()
        t.execute()
        return t

    def forget(self, column: existing_column):
        column = self.get_existing_column(column)
        self.verify_columns([column], {Table.ColumnRequirements.IS_VAL})
        t = Table.create_from_table(self)
        idx = find_index(column.name, self.displayed_columns)
        if idx < t.marker:
            t.marker -= 1
        t.set_displayed_columns(t.displayed_columns[:idx] + t.displayed_columns[idx + 1:])
        t.derivation = self.derivation.forget(column)
        t.extend_intermediate_representation()
        t.execute()
        return t

    def show(self, column: existing_column):
        if isinstance(column, str):
            name = column
        else:
            name = column.name
        t = Table.create_from_table(self)
        col = t.derivation.find_hidden(name)
        t.set_displayed_columns(t.displayed_columns[:t.marker] + [column] + t.displayed_columns[t.marker:])
        t.marker += 1
        t.derivation = self.derivation.show(col)
        t.extend_intermediate_representation()
        t.execute()
        return t

    def invert(self, keys: list[existing_column], vals: list[existing_column]):
        t = Table.create_from_table(self)

        keys = t.get_existing_columns(keys)
        vals = t.get_existing_columns(vals)

        all_keys = t.derivation.get_keys()
        key_cols = list(sorted(all_keys, key=lambda c: find_index(c.name, self.displayed_columns)))

        all_vals = t.derivation.get_values()
        val_cols = list(sorted(all_vals, key=lambda c: find_index(c.name, self.displayed_columns)))

        key_doms = [c.get_domain() for c in keys]
        val_doms = [c.get_domain() for c in vals]

        key_node = t.derivation.find_node_with_domains(key_doms)
        val_node = t.derivation.find_node_with_domains(val_doms)

        path = key_node.path_to_value(val_node)

        assert path is not None
        assert len(path) > 1

        indices_to_replace = [find_index(k, key_cols) for k in keys]
        idx = indices_to_replace[0]

        all_keys = t.derivation.domains

        filtered_keys = [k for i, k in enumerate(all_keys) if i not in set(indices_to_replace)]
        new_keys = filtered_keys[:idx] + val_doms + filtered_keys[idx:]

        visible_key_indices_to_replace = [find_index(k.name, t.displayed_columns) for k in keys]
        visible_val_indices_to_replace = [find_index(v.name, t.displayed_columns) for v in vals]
        visible_indices_to_replace = visible_key_indices_to_replace + visible_val_indices_to_replace
        visible_idx = visible_indices_to_replace[0]
        filtered_displayed_columns = [c for i, c in enumerate(t.displayed_columns) if
                                      i not in set(visible_indices_to_replace)]
        t.displayed_columns = (filtered_displayed_columns[:visible_idx] + [v.name for v in val_doms] +
                               filtered_displayed_columns[visible_idx:] + [k.name for k in key_doms])
        t.marker += len(val_doms) - np.sum(np.array(visible_key_indices_to_replace) < t.marker)
        # Let k be the key we're inverting
        old_root = t.derivation

        key_node.parent = None

        new_root = DerivationNode.create_root(new_keys)

        res, inverted_repr = DerivationNode.invert_path(path, t)

        children = old_root.children.to_list()
        groups = classify_groups(children, key_doms)

        # Six groups
        # Holy fuck

        # Group 0 {unit key}
        group0 = groups[0]
        assert len(group0) <= 1
        # Group 4 {k' | k and k' disjoint}
        group4 = groups[4]

        for c in group0 + group4:
            new_root = new_root.insert_key(c.domains)
            key = new_root.find_node_with_domains(c.domains)
            new_root = new_root.add_children(key, c.children)

        # todo: update cardinality
        new_root = new_root.insert_key(val_doms)
        new_key = new_root.find_node_with_domains(val_doms)
        new_root = new_root.add_child(new_key, res)

        # Group 1 {k' | k' is subset of k}
        group1 = groups[1]
        for child in group1:
            # if child in set(key_node.children.item_list):
            #     key_node.remove_child_node(child)
            if len(child.domains) > 1:
                intermediate = child.to_intermediate_node()
                intermediate = intermediate.add_children(intermediate, child.children)
                new_root = new_root.add_child(new_key, intermediate)
            else:
                value = child.to_value_column()
                value = value.add_children(value, child.children)
                new_root = new_root.add_child(new_key, value)

        to_merge = {}

        # Group 2 {k' | k is subset of k'}
        group2 = groups[2]

        for child in group2:
            difference = [d for d in child.domains if d not in set(key_node.children.item_list)]
            assert difference not in to_merge
            child.parent = None
            to_merge[difference] = child
            new_root = new_root.insert_key(val_doms + difference)
            possible = child.find_node_with_domains(val_doms + difference)
            key_node = new_root.find_node_with_domains(val_doms + difference)
            if possible is None:
                intermediate = child.to_intermediate_node()
                intermediate = intermediate.add_children(intermediate, child.children)
                new_ir = carry_keys_through_representation(inverted_repr, difference)
                intermediate.internal_representation = new_ir
                new_root = new_root.add_child(key_node, intermediate)
            else:
                path = child.path_to_value(val_doms + difference)
                inverted_path = DerivationNode.invert_path(path, t)
                new_root = new_root.add_child(key_node, inverted_path[1])
                penultimate = new_root.find_node_with_domains(inverted_path[-2].domains)
                end = new_root.find_node_with_domains(child.domains)
                intermediate = end.to_intermediate_node()
                intermediate = intermediate.add_children(intermediate, inverted_path[-1].children)
                new_root = new_root.remove_child(end).add_child(penultimate, intermediate)
                to_merge[difference] = intermediate

        # Group 3 {k' | k and k' overlap}
        group3 = groups[3]

        for child in group3:
            difference = [d for d in child.domains if d not in set(key_node.children.item_list)]
            if difference in to_merge:
                new_root = new_root.add_child(to_merge[difference], child)
            else:
                new_root = new_root.insert_key(val_doms + difference)
                key_node = new_root.find_node_with_domains(val_doms + difference)
                new_ir = carry_keys_through_representation(inverted_repr, difference)
                intermediate = IntermediateNode(key_doms + difference, new_ir, res.hidden_keys.to_list())
                new_root = new_root.add_child(key_node, intermediate)
                new_root = new_root.add_child(intermediate, child)
                to_merge[difference] = intermediate

        t.derivation = new_root
        t.extend_intermediate_representation()
        t.execute()
        return t

    def shift_left(self):
        assert self.marker > 1
        t = Table.create_from_table(self)
        col = t.get_existing_column(t.displayed_columns[t.marker - 1])
        if col.is_key_column():
            new_name = t.get_fresh_name(col.name, t.get_namespace())
            temp_name = t.get_fresh_name(col.name, t.get_namespace() | {new_name})
            t = t.infer([col], col.get_domain().node, with_name=temp_name)
            t = t.rename(col.name, new_name).rename(temp_name, col.name)
            t = t.hide(new_name)
            t.displayed_columns = t.displayed_columns[:t.marker] + [col.name] + t.displayed_columns[t.marker:-1]
        else:
            t.marker -= 1
        t.extend_intermediate_representation()
        t.execute()
        return t

    def shift_right(self):
        assert self.marker <= len(self.displayed_columns) - 1
        t = Table.create_from_table(self)
        t.marker += 1
        t.extend_intermediate_representation()
        t.execute()
        return t

    def equate(self, col1, col2):
        self.verify_columns([col1, col2], {Table.ColumnRequirements.IS_KEY})

        col1 = self.get_existing_column(col1)
        col2 = self.get_existing_column(col2)

        t = Table.create_from_table(self)
        idx = self.displayed_columns.index(col2.name)
        if idx < t.marker:
            t.marker -= 1
        t.set_displayed_columns([c for c in self.displayed_columns if c != col2.name])
        t.derivation = self.derivation.equate(col1, col2)
        t.extend_intermediate_representation()
        t.execute()
        return t

    def swap(self, left: str, right: str):
        t = Table.create_from_table(self)
        i = find_index(left, t.displayed_columns)
        j = find_index(right, t.displayed_columns)
        assert i+1 == j
        # TODO: check that left not strong key for right
        # TODO: modify derivation tree if both keys
        t.displayed_columns = t.displayed_columns[:i] + [t.displayed_columns[j], t.displayed_columns[i]] + t.displayed_columns[j+1:]
        t.extend_intermediate_representation()
        t.execute()
        return t

    def filter(self, by: existing_column):
        column = self.get_existing_column(by)
        self.verify_columns([column], {Table.ColumnRequirements.IS_KEY_OR_VAL})
        t = Table.create_from_table(self)
        t.extend_intermediate_representation([Filter(column.get_domain())])
        t.execute()
        return t

    def sort(self, cols: list[str]):
        t = Table.create_from_table(self)
        t.extend_intermediate_representation([Sort(cols)])
        t.execute()
        return t

    def rename(self, old_name: str, new_name: str):
        assert new_name not in self.get_namespace()
        t = Table.create_from_table(self)
        idx = find_index(old_name, t.displayed_columns)
        assert idx >= 0
        t.derivation = self.derivation.rename(old_name, new_name)
        t.displayed_columns = t.displayed_columns[:idx] + [new_name] + t.displayed_columns[idx+1:]
        t.extend_intermediate_representation()
        t.execute()
        return t

    def __repr__(self):
        left, _, right = self.get_columns_as_lists()
        left = ' '.join([l.get_name() for l in left])
        right = ' '.join([r.get_name() for r in right])
        dropped_keys = f"\n{self.dropped_keys_count} keys hidden"
        dropped_vals = f"\n{self.dropped_vals_count} values hidden"
        repr = f"[{left} || {right}]" + "\n" + str(self.df)
        if self.dropped_keys_count > 0:
            repr += dropped_keys
        if self.dropped_vals_count > 0:
            repr += dropped_vals
        return repr + "\n\n"

    def __str__(self):
        return self.__repr__()

    def __getitem__(self, item):
        col = self.derivation.find_column_with_name(item)
        if col is not None:
            return Column(col)
        raise KeyError
