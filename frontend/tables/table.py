from __future__ import annotations

import copy
import uuid
from enum import Enum
from typing import Any

import numpy as np

from backend.populated_table import PopulatedTable
from exp.exp import Exp
from frontend.derivation.derivation_node import (
    DerivationNode,
    ColumnNode,
    IntermediateNode,
    RootNode,
    invert_derivation_path,
    intermediate_representation_for_path,
)
from representation.domain import Domain
from frontend.tables.column import Column
from frontend.tables.exceptions import (
    ColumnsNeedToBeUniqueException,
    ColumnsNeedToBeInTableAndVisibleException,
    ColumnsNeedToBeKeysException,
    ColumnsNeedToBeInTableException,
    ColumnsNeedToBeValuesException,
)
from frontend.tables.helpers.carry_keys_through_representation import (
    carry_keys_through_representation,
)
from frontend.tables.helpers.flatten import flatten
from representation.representation import (
    RepresentationStep,
    End,
    Filter,
    Sort,
    Get,
    EndTraversal,
    StartTraversal,
    Pop,
)
from schema.cardinality import Cardinality
from schema.edge import SchemaEdge
from schema.helpers.compose_cardinality import compose_cardinality
from schema.helpers.find_index import find_index
from schema.helpers.is_sublist import is_sublist
from schema.node import SchemaNode, AtomicNode, SchemaClass

existing_column = str | Column | Domain | ColumnNode
new_column = AtomicNode | SchemaClass | Column | Domain | ColumnNode


def classify_groups(
    keys: list[DerivationNode], to_invert: list[Domain]
) -> dict[int, list[DerivationNode]]:
    """
    Classifies the keys of a derivation tree into groups based on their domains and the domains to invert
    The groups are:
        0: Keys with no domains
        1: Keys with domains that are a subset of the domains to invert
        2: Keys with domains that are a superset of the domains to invert
        3: Keys with domains that intersect with the domains to invert, but are neither a subset nor superset
        4: Keys with domains that are completely disjoint with the domains to invert

    Args:
        keys (list[DerivationNode]): a list of key nodes
        to_invert (list[Domain]): the domains to invert
    """
    groups = {n: [] for n in range(5)}
    for i, child in enumerate(keys):
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


def get_name_and_node(i: new_column) -> tuple[str, AtomicNode]:
    """
    Extracts the name and node from a column or a schema node
    Args:
        i: a column or a schema node
    Returns:
        a tuple of the name and the node
    """

    if isinstance(i, AtomicNode | SchemaClass):
        return i.name, i
    elif isinstance(i, Column):
        return i.name, i.get_schema_node()
    elif isinstance(i, Domain):
        return i.name, i.node
    elif isinstance(i, ColumnNode):
        return i.get_name(), i.get_schema_node()


def get_names_and_nodes(input_columns: list[new_column]) -> tuple[list[Any], ...]:
    """
    Extracts the names and nodes from a list of columns or schema nodes

    Args:
        input_columns: a list of columns or schema nodes

    Returns:
        A tuple of lists of names and nodes
    """
    return tuple(
        [list(x) for x in zip(*[get_name_and_node(ic) for ic in input_columns])]
    )


def get_fresh_name(name: str, namespace: frozenset[str]) -> str:
    """
    Generates a fresh name by appending a number to the name if it already exists in the namespace

    Args:
        name: the name
        namespace: the namespace

    Returns:
        The fresh name
    """
    candidate = name
    if candidate in namespace:
        i = 1
        candidate = f"{name}_{i}"
        while candidate in namespace:
            i += 1
            candidate = f"{name}_{i}"
    return candidate


def new_domain_from_schema_node(
    namespace: frozenset[str], node: AtomicNode, name: str = None
):
    """
    Creates a new domain from a schema node, such that the name of the domain is not already in the namespace

    Args:
        namespace: the namespace
        node: the schema node
        name: the name of the column

    Returns:
        The new domain
    """
    if name is None:
        name = get_fresh_name(node.name, namespace)
    else:
        name = get_fresh_name(name, namespace)
    return Domain(name, node)


def create_domain(name: str, node: AtomicNode, namespace: frozenset[str]) -> Domain:
    """
    Creates a domain from a name and a schema node

    Args:
        name: the name of the domain
        node: the schema node
        namespace: the namespace of the table

    Returns:
        The domain
    """
    name = get_fresh_name(name, namespace)
    return Domain(name, node)


class Table:
    """
    A table is a collection of columns, and a derivation tree that
    describes how to derive each value from a subset of the keys.
    """

    class ColumnRequirements(Enum):
        IS_KEY = 1
        IS_HIDDEN = 2
        IS_VAL = 3
        IS_KEY_OR_VAL = 4
        IS_KEY_OR_HIDDEN = 5
        IS_VAL_OR_HIDDEN = 6
        IS_KEY_OR_VAL_OR_HIDDEN = 7
        IS_UNIQUE = 8

    def __init__(
        self,
        table_id,
        derivation: RootNode | None,
        intermediate_representation: list[RepresentationStep],
        schema,
        derived_from=None,
    ):
        self.table_id = table_id
        self.derived_from = derived_from
        self.displayed_columns = []
        self.marker = 0  # index of the first value column
        self.intermediate_representation = intermediate_representation
        self.derivation = derivation
        self.schema = schema
        self.populated_table: PopulatedTable | None = None

    @classmethod
    def construct(cls, columns: list[Domain], schema: "Schema") -> Table:
        """
        Constructs a table from a list of columns and a schema

        Args:
            columns (list[Domain]): a list of columns
            schema (Schema): the schema

        Returns:
            Table: the constructed table
        """
        table_id = uuid.uuid4().hex
        table = Table(table_id, None, [], schema)
        keys = []
        namespace = frozenset()
        for column in columns:
            key_node = column.node
            name = column.name
            key = create_domain(name, key_node, namespace)
            keys += [key]
            namespace |= {name}

        table.displayed_columns = [str(k) for k in keys]
        table.left = {str(k): k for k in keys}
        table.populated_table = None
        table.marker = len(keys)
        table.derivation = DerivationNode.create_root(keys)
        table.execute()

        return table

    @classmethod
    def create_from_table(cls, table: Table) -> Table:
        """
        Creates a new table from an existing table

        Args:
            table: the existing table

        Returns:
            The new table
        """
        table_id = uuid.uuid4().hex
        new_table = Table(
            table_id,
            table.derivation,
            table.intermediate_representation,
            table.schema,
            table.table_id,
        )
        new_table.displayed_columns = copy.copy(table.displayed_columns)
        new_table.marker = table.marker
        new_table.populated_table = table.populated_table.copy()
        return new_table

    def __get_existing_column(self, input: existing_column) -> ColumnNode:
        """
        Users may specify a column in the table as a column, column node, or string
        Extracts the column node from a column, column node, or string

        Args:
            input: a column, column node, or string

        Returns:
            The column node
        """
        if isinstance(input, str):
            return self.derivation.find_column_with_name(input)
        elif isinstance(input, Column):
            return input.node
        elif isinstance(input, Domain):
            return self.derivation.find_column_with_name(input.name)
        elif isinstance(input, ColumnNode):
            return input

    def __get_existing_columns(self, inputs: list[existing_column]) -> list[ColumnNode]:
        """
        Extracts the column nodes from a list of user-specified columns

        Args:
            inputs: a list of user-specified columns

        Returns:
            A list of column nodes
        """
        columns = []
        for input in inputs:
            columns += [self.__get_existing_column(input)]
        return columns

    def execute(
        self, with_new_representation: list[RepresentationStep] | None = None
    ) -> None:
        """
        Executes the intermediate representation to populate the table

        Args:
            with_new_representation: a list of new representation steps to append to the existing intermediate representation
        """
        if with_new_representation is None:
            with_new_representation = []
        self.intermediate_representation = (
            self.derivation.to_intermediate_representation()
        )
        if len(self.intermediate_representation) > 0:
            self.intermediate_representation = (
                self.intermediate_representation + with_new_representation
            )
        else:
            self.intermediate_representation = with_new_representation
        left, hids, right = self.get_columns_as_lists()
        self.intermediate_representation += [End(left, hids, right)]
        print(self.derivation)

        self.populated_table, self.schema = self.schema.execute_query(
            self.table_id, self.derived_from, self.intermediate_representation
        )

    def verify_columns(
        self, columns: list[ColumnNode], requirements: set[ColumnRequirements]
    ):
        """
        Verifies that a list of columns satisfies a set of requirements

        Args:
            columns: a list of columns
            requirements: a set of requirements

        Raises:
            ColumnsNeedToBeUniqueException: if the columns are not unique
            ColumnsNeedToBeInTableException: if the columns are not in the table
            ColumnsCannotAlreadyBeInTableException: if the columns are already in the table
            ColumnsNeedToBeKeysException: if the columns are not keys
            ColumnsNeedToBeValuesException: if the columns are not values
            ColumnsNeedToBeHiddenException: if the columns are not hidden
        """
        keys = set(self.derivation.get_keys())
        vals = set(self.derivation.get_values())
        hids = set(self.derivation.get_hidden())
        if Table.ColumnRequirements.IS_UNIQUE in requirements and len(
            set(columns)
        ) != len(columns):
            raise ColumnsNeedToBeUniqueException()
        if Table.ColumnRequirements.IS_KEY_OR_VAL_OR_HIDDEN in requirements and not set(
            columns
        ).issubset(keys | vals | hids):
            raise ColumnsNeedToBeInTableException()
        if Table.ColumnRequirements.IS_KEY_OR_VAL in requirements and not set(
            columns
        ).issubset(keys | vals):
            raise ColumnsNeedToBeInTableAndVisibleException()
        if Table.ColumnRequirements.IS_KEY in requirements and not set(
            columns
        ).issubset(keys):
            raise ColumnsNeedToBeKeysException()
        if Table.ColumnRequirements.IS_VAL in requirements and not set(
            columns
        ).issubset(vals):
            raise ColumnsNeedToBeValuesException()
        if Table.ColumnRequirements.IS_UNIQUE in requirements and not len(
            set(columns)
        ) == len(columns):
            raise ColumnsNeedToBeUniqueException()

    def get_domain_with_name(self, name: str) -> Domain | None:
        """
        Gets the domain with a given name

        Args:
            name: the name of the domain

        Returns:
            The domain with the given name, or None if it does not exist
        """
        return self.derivation.find_column_with_name(name)

    def get_columns_as_lists(
        self,
    ) -> tuple[list[ColumnNode], list[ColumnNode], list[ColumnNode]]:
        """
        Gets the columns in the table as lists

        Returns:
            A tuple of lists of columns
            left: columns to the left of the marker
            hids: columns that are hidden
            right: columns to the right of the marker
        """
        keys_and_values = self.derivation.get_keys_and_values()
        visible = list(
            sorted(
                filter(lambda c: find_index(c.name, self.displayed_columns) >=0, keys_and_values),
                key=lambda c: find_index(c.name, self.displayed_columns),
            )
        )
        left = visible[: self.marker]
        right = visible[self.marker :]
        hids = self.derivation.get_hidden()
        return left, hids, right

    def set_displayed_columns(self, new_columns: list[str]) -> None:
        """
        Sets the displayed columns

        Args:
            new_columns: the new columns
        """
        self.displayed_columns = new_columns

    def find_strong_keys_for_columns(self, columns: list[ColumnNode]) -> list[Domain]:
        """
        Finds the strong keys for a list of columns

        Args:
            columns: a list of columns

        Returns:
            A list of strong keys
        """
        strong_keys = set()
        for column in columns:
            strong_keys_of_column = column.get_strong_keys()
            if column.is_key_column():
                strong_keys = strong_keys.union([column.get_domain()])
            else:
                strong_keys = strong_keys.union(strong_keys_of_column)

        return list(
            sorted(
                strong_keys, key=lambda x: find_index(x.name, self.displayed_columns)
            )
        )

    def find_hidden_keys_for_column(
        self, columns: list[ColumnNode], aggregated_over: set[ColumnNode]
    ) -> list[Domain]:
        """
        Finds the hidden keys for a list of columns

        Args:
            columns: a list of columns
            aggregated_over: a set of columns, in the list of columns, that are aggregated over

        Returns:
            A list of hidden keys
        """
        hidden_keys = set()
        blocked_hidden_keys = self.blocked_hidden_keys()
        for column in columns:
            if column.is_hidden_key_column():
                hidden_keys_of_column = [column.get_domain()]
            else:
                hidden_keys_of_column = column.get_hidden_keys()
            if column.get_domain() in aggregated_over:
                hidden_keys = hidden_keys.union(
                    set(hidden_keys_of_column).intersection(blocked_hidden_keys)
                )
            else:
                hidden_keys = hidden_keys.union(hidden_keys_of_column)
        return list(
            sorted(
                hidden_keys,
                key=lambda x: find_index(
                    x, [c.get_domain() for c in self.derivation.get_hidden()]
                ),
            )
        )

    def get_namespace(self) -> frozenset[str]:
        """
        Gets the namespace of the table

        Returns:
            The namespace
        """
        return frozenset(
            [
                d.name
                for d in self.derivation.get_keys_and_values()
                + self.derivation.get_hidden()
            ]
        )

    def get_representation(
        self, start: list[Domain], end: list[Domain], via, namespace
    ):

        from frontend.tables.helpers.convert_path_to_representation import (
            convert_path_to_representation,
        )

        cardinality, edges = self.schema.find_shortest_path(start, end, via)
        new_repr, hidden_columns = convert_path_to_representation(
            start, end, edges, namespace
        )
        return cardinality, new_repr, hidden_columns

    def compose(
        self,
        from_keys: list[new_column],
        to_key: existing_column,
        via: list[str] = None,
    ):
        from_keys_names, from_keys_nodes = get_names_and_nodes(from_keys)
        self.verify_columns(from_keys_names, {Table.ColumnRequirements.IS_UNIQUE})
        key = self.__get_existing_column(to_key)
        self.verify_columns([key], {Table.ColumnRequirements.IS_KEY})

        key_idx = find_index(to_key, self.displayed_columns)

        # STEP 1
        # Initialise the new table
        # Delete the key we want to compose on from its namespace
        t = Table.create_from_table(self)

        namespace = t.get_namespace()
        namespace = namespace.difference({to_key})
        # STEP 2
        # 2a. Update the columns to display

        cols_to_add = []
        for k in from_keys_nodes:
            col = new_domain_from_schema_node(namespace, k)
            namespace |= {col.name}
            cols_to_add += [col]

        t.set_displayed_columns(
            self.displayed_columns[:key_idx]
            + [str(c) for c in cols_to_add]
            + self.displayed_columns[key_idx + 1 :]
        )
        # 2b. Update the marker
        t.marker = self.marker + len(from_keys) - 1

        # STEP 3
        # Determine the derivation path
        # Want a (backwards) path from from_keys <------- to_key
        # Create start, via, and end nodes
        via_nodes = None
        if via is not None:
            via_nodes = [t.schema.get_node_with_name(n) for n in via]
        shortest_p = t.get_representation(
            cols_to_add, [key.get_domain()], via_nodes, namespace
        )
        cardinality, repr, hidden_columns = shortest_p

        # STEP 4
        # compute the new intermediate representation
        t.derivation = self.derivation.compose(
            cols_to_add, key.get_domain(), hidden_columns, repr, cardinality
        )
        t.execute()

        return t

    def blocked_hidden_keys(self):
        left, _, _ = self.get_columns_as_lists()
        hidden = set(flatten([c.get_hidden_keys() for c in left]))
        return hidden

    def mutate(self, **kwargs):
        t = Table.create_from_table(self)
        for name, expr in kwargs.items():
            t = t.deduce(expr, name)
        return t

    def deduce(self, function: Exp, with_name: str):
        t = Table.create_from_table(self)
        exp, start_columns, aggregated_over, usages = Exp.convert_exp(function)
        start_columns = [c for c in start_columns]
        start_node = SchemaNode.product([c.node for c in start_columns])

        node_type = function.exp_type
        end_node = self.schema.add_node(AtomicNode(with_name, node_type))
        modified_start_node = None
        modified_start_cols = None
        if len(aggregated_over) > 0:
            modified_start_cols = [
                i
                for i, c in enumerate(start_columns)
                if c.name in t.displayed_columns
                and (
                    i not in aggregated_over
                    or i not in usages
                    or aggregated_over[i] != usages[i]
                )
            ]
            if len(modified_start_cols) > 0:
                modified_start_node = SchemaNode.product(
                    [start_columns[i].node for i in modified_start_cols]
                )

                if not self.schema.does_edge_exist_in_graph(
                    modified_start_node, end_node
                ):
                    self.schema.add_edge(
                        modified_start_node, end_node, Cardinality.MANY_TO_ONE
                    )
                if not self.schema.does_edge_exist_in_graph(
                    end_node, modified_start_node
                ):
                    self.schema.add_edge(
                        end_node, modified_start_node, Cardinality.ONE_TO_MANY
                    )
        else:
            if not self.schema.does_edge_exist_in_graph(start_node, end_node):
                self.schema.add_edge(start_node, end_node, Cardinality.MANY_TO_ONE)

        if modified_start_node is None:
            modified_start_node = start_node
        if modified_start_cols is None:
            modified_start_cols = [i for i in range(len(start_columns))]
        edge = SchemaEdge(modified_start_node, end_node, Cardinality.MANY_TO_ONE)
        data = self.populated_table.evaluate_exp(
            exp, start_columns, modified_start_cols
        )
        self.schema.map_edge_to_data(edge, data)
        self.schema.map_atomic_node_to_data(end_node, data[len(modified_start_cols)])
        t_new = self
        assumption = [
            start_columns[i].name
            for i in modified_start_cols
            if start_columns[i].name in t_new.displayed_columns
        ]
        t_new = t_new.infer_internal(assumption, end_node, with_name=with_name)
        return t_new

    def infer(
        self,
        from_columns: list[existing_column],
        to_column: new_column,
        via: list[SchemaNode] = None,
        with_name: str = None,
    ):

        assumption_columns = self.__get_existing_columns(from_columns)

        self.verify_columns(
            assumption_columns,
            {
                Table.ColumnRequirements.IS_KEY_OR_VAL,
                Table.ColumnRequirements.IS_UNIQUE,
            },
        )

        return self.infer_internal(assumption_columns, to_column, via, with_name)

    def infer_internal(
        self,
        from_columns: list[existing_column],
        to_column: new_column,
        via: list[SchemaNode] = None,
        with_name: str = None,
        aggregated_over: list[Domain] = None,
    ):
        # An inference from a set of assumption columns to a conclusion column
        assumption_columns = self.__get_existing_columns(from_columns)
        self.verify_columns(
            assumption_columns,
            {
                Table.ColumnRequirements.IS_KEY_OR_VAL_OR_HIDDEN,
                Table.ColumnRequirements.IS_UNIQUE,
            },
        )

        to_column_name, to_column_node = get_name_and_node(to_column)
        if with_name is not None:
            to_column_name = with_name

        t = Table.create_from_table(self)

        namespace = t.get_namespace()

        # STEP 1
        # Get name for inferred column
        # Append column to end of displayed columns
        name = get_fresh_name(to_column_name, namespace)
        t.set_displayed_columns(t.displayed_columns + [name])

        namespace |= {name}

        strong_keys = t.find_strong_keys_for_columns(assumption_columns)

        if isinstance(to_column, Column) or isinstance(to_column, ColumnNode):
            if isinstance(to_column, Column):
                col = to_column.node
            else:
                col = to_column
            root = col.find_root_of_tree()

            domains = []
            for from_col, assum_col in zip(col.get_strong_keys(), assumption_columns):
                # from_col = from_col.column

                assum_col = assum_col.get_domain()
                if from_col != assum_col:
                    ns = set([d.name for d in root.get_keys_and_values() + root.get_hidden()])
                    cardinality, repr, hidden_columns = self.get_representation([assum_col], [from_col], [], namespace=ns)
                    root = root.compose([assum_col], from_col, hidden_columns, repr, cardinality=cardinality)
                    domains += [Domain(assum_col.name, assum_col.node)]
                else:
                    domains += [Domain(from_col.name, from_col.node)]

            if to_column.name != to_column_name:
                root = root.rename(to_column.name, to_column_name)
            key_node = root.find_node_with_domains(domains)
            val_node = root.find_node_with_domains(
                [Domain(to_column_name, to_column_node)]
            )
            path = key_node.path_to_value(val_node)

            t.derivation = self.derivation.splice(path, self.get_namespace())
            t.execute()
            return t

            # STEP 2
        # Get the shortest path in the schema graph
        if aggregated_over is None:
            aggregated_over = []
        old_hidden_keys = t.find_hidden_keys_for_column(
            assumption_columns, set(aggregated_over)
        )
        cardinalities = [column.cardinality for column in assumption_columns]
        conclusion_column = Domain(name, to_column_node)

        if len(assumption_columns) == 0:
            pass
            strong_keys = []
            hidden_key = Domain(get_fresh_name(name, namespace), to_column_node)
            repr = [
                Pop(),
                Get([hidden_key]),
                StartTraversal([hidden_key]),
                EndTraversal([conclusion_column]),
            ]
            hidden_keys = [hidden_key]
            cardinality = Cardinality.MANY_TO_MANY
        else:
            via_nodes = None
            if via is not None:
                via_nodes = via

            shortest_p = t.get_representation(
                [c.get_domain() for c in assumption_columns],
                [conclusion_column],
                via_nodes,
                namespace,
            )
            cardinality, repr, hidden_keys = shortest_p

        # 2b. Turn the hidden keys into columns, and rename them if necessary.
        hidden_assumptions = [hk if str(hk) != name else hk for hk in hidden_keys]

        # STEP 3.
        # Update keys, vals, and hidden keys (keys don't change)
        for c in cardinalities:
            if c is not None:
                cardinality = compose_cardinality(c, cardinality)

        new_col = Domain(name, to_column_node)

        # TODO: Prepend
        # STEP 4
        # Update intermediate representation
        t.derivation = self.derivation.infer(
            new_col,
            strong_keys,
            old_hidden_keys,
            hidden_assumptions,
            [c for c in assumption_columns],
            cardinality,
            repr,
        )
        t.execute()

        return t

    def hide(self, column: existing_column) -> Table:
        column = self.__get_existing_column(column)
        self.verify_columns([column], {Table.ColumnRequirements.IS_KEY_OR_VAL})
        t = Table.create_from_table(self)
        idx = find_index(column.name, self.displayed_columns)
        if idx < t.marker:
            t.marker -= 1
        t.set_displayed_columns(
            t.displayed_columns[:idx] + t.displayed_columns[idx + 1 :]
        )
        t.derivation = self.derivation.hide(column)
        t.execute()
        return t

    def show(self, column: existing_column) -> Table:
        """
        Shows a hidden column
        """
        if isinstance(column, str):
            name = column
        else:
            name = column.name
        t = Table.create_from_table(self)
        col = t.derivation.find_column_with_name(name).get_domain()
        t.set_displayed_columns(
            t.displayed_columns[: t.marker] + [column] + t.displayed_columns[t.marker :]
        )
        t.marker += 1
        t.derivation = self.derivation.show(col)
        t.execute()
        return t

    def group_by(self, columns: list[existing_column]) -> Table:
        """
        Groups the table by a list of columns
        """

        columns = self.__get_existing_columns(columns)
        self.verify_columns(columns, {Table.ColumnRequirements.IS_KEY_OR_VAL})
        all_keys_and_vals = self.derivation.get_keys_and_values()
        keys = [c for c in columns if c.is_key_column()]
        key_doms = [c.get_domain() for c in keys]
        vals = [c for c in columns if c.is_val_column()]
        col_doms = [c.get_domain() for c in columns]
        keys_to_invert_set = set()
        keys_to_invert = {}
        the_rest = set(all_keys_and_vals).difference(set(keys) | set(vals))

        for val in vals:
            keys_of_vals = val.get_strong_keys()
            grouped_keys = set(keys_of_vals).intersection(set(key_doms))
            remaining_keys = [k for k in keys_of_vals if k not in grouped_keys]
            the_rest = [c for c in the_rest if c.get_domain() not in remaining_keys]
            for k in remaining_keys:
                if k not in keys_to_invert:
                    keys_to_invert[k] = set()
                keys_to_invert[k] |= set(grouped_keys).union({val.get_domain()})
            keys_to_invert_set |= set(remaining_keys)

        keys_to_invert = {
            k: list(sorted(list(v), key=lambda x: find_index(x, col_doms)))
            for k, v in keys_to_invert.items()
        }
        keys_to_invert_list = list(
            sorted(
                list(keys_to_invert_set),
                key=lambda x: find_index(x.name, self.displayed_columns),
            )
        )
        the_rest = list(
            sorted(
                list(the_rest), key=lambda x: find_index(x.name, self.displayed_columns)
            )
        )

        t = Table.construct(col_doms, self.schema)

        for k in keys_to_invert_list:
            strong_keys = keys_to_invert[k]
            data = self.populated_table.group_by(strong_keys, k)
            # TODO: clone
            start_node = SchemaNode.product([c.node for c in strong_keys])
            end_node = k.node
            edge = SchemaEdge(start_node, end_node, Cardinality.ONE_TO_MANY)
            self.schema.add_edge(start_node, end_node, Cardinality.ONE_TO_MANY)
            self.schema.map_edge_to_data(edge, data)
            t = t.infer(strong_keys, k)

        for c in the_rest:
            strong_keys = c.get_strong_keys()
            t = t.infer(strong_keys, c)

        return t

    ## TODO: Go through with Damon?
    def invert(self, keys: list[existing_column], vals: list[existing_column]) -> Table:
        """
        Inverts the keys and values

        Args:
            keys: the keys
            vals: the values
        """
        t = Table.create_from_table(self)

        keys = t.__get_existing_columns(keys)
        vals = t.__get_existing_columns(vals)

        all_keys = t.derivation.get_keys()
        key_cols = list(
            sorted(all_keys, key=lambda c: find_index(c.name, self.displayed_columns))
        )

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

        filtered_keys = [
            k for i, k in enumerate(all_keys) if i not in set(indices_to_replace)
        ]
        new_keys = filtered_keys[:idx] + val_doms + filtered_keys[idx:]

        visible_key_indices_to_replace = [
            find_index(k.name, t.displayed_columns) for k in keys
        ]
        visible_val_indices_to_replace = [
            find_index(v.name, t.displayed_columns) for v in vals
        ]
        visible_indices_to_replace = (
            visible_key_indices_to_replace + visible_val_indices_to_replace
        )
        visible_idx = visible_indices_to_replace[0]
        filtered_displayed_columns = [
            c
            for i, c in enumerate(t.displayed_columns)
            if i not in set(visible_indices_to_replace)
        ]
        t.displayed_columns = (
            filtered_displayed_columns[:visible_idx]
            + [v.name for v in val_doms]
            + filtered_displayed_columns[visible_idx:]
            + [k.name for k in key_doms]
        )
        t.marker += len(val_doms) - np.sum(
            np.array(visible_key_indices_to_replace) < t.marker
        )
        # Let k be the key we're inverting
        old_root = t.derivation

        key_node.parent = None

        new_root = DerivationNode.create_root(new_keys)

        res = invert_derivation_path(path, t.get_namespace())
        res = res.children[0]
        new_path = res.path_to_value(key_node)
        inverted_repr = intermediate_representation_for_path(new_path)

        children = old_root.children.to_list()
        groups = classify_groups(children, key_doms)

        # Six groups

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
            difference = [
                d for d in child.domains if d not in set(key_node.children.item_list)
            ]
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
                inverted_path = invert_derivation_path(path, t.get_namespace())
                new_root = new_root.add_child(key_node, inverted_path[1])
                penultimate = new_root.find_node_with_domains(inverted_path[-2].domains)
                end = new_root.find_node_with_domains(child.domains)
                intermediate = end.to_intermediate_node()
                intermediate = intermediate.add_children(
                    intermediate, inverted_path[-1].children
                )
                new_root = new_root.remove_child(end).add_child(
                    penultimate, intermediate
                )
                to_merge[difference] = intermediate

        # Group 3 {k' | k and k' overlap}
        group3 = groups[3]

        for child in group3:
            difference = [
                d for d in child.domains if d not in set(key_node.children.item_list)
            ]
            if difference in to_merge:
                new_root = new_root.add_child(to_merge[difference], child)
            else:
                new_root = new_root.insert_key(val_doms + difference)
                key_node = new_root.find_node_with_domains(val_doms + difference)
                new_ir = carry_keys_through_representation(inverted_repr, difference)
                intermediate = IntermediateNode(
                    key_doms + difference, new_ir, res.hidden_keys.to_list()
                )
                new_root = new_root.add_child(key_node, intermediate)
                new_root = new_root.add_child(intermediate, child)
                to_merge[difference] = intermediate

        t.derivation = new_root
        t.execute()
        return t

    def shift_left(self) -> Table:
        """
        Shifts the marker to the left

        Returns:
            The new table
        """
        assert self.marker > 1
        t = Table.create_from_table(self)
        col = t.__get_existing_column(t.displayed_columns[t.marker - 1])
        if col.is_key_column():
            new_name = get_fresh_name(col.name, t.get_namespace())
            temp_name = get_fresh_name(col.name, t.get_namespace() | {new_name})
            t = t.infer([col], col.get_domain().node, with_name=temp_name)
            t = t.rename(col.name, new_name).rename(temp_name, col.name)
            t = t.hide(new_name)
            t.displayed_columns = (
                t.displayed_columns[: t.marker]
                + [col.name]
                + t.displayed_columns[t.marker : -1]
            )
        else:
            t.marker -= 1
        t.execute()
        return t

    def shift_right(self) -> Table:
        """
        Shifts the marker to the right

        Returns:
            The new table
        """
        assert self.marker <= len(self.displayed_columns) - 1
        t = Table.create_from_table(self)
        t.marker += 1
        t.execute()
        return t

    def equate(self, col1: existing_column, col2: existing_column) -> Table:
        """
        Equates two columns in the table

        Args:
            col1: the first column
            col2: the second column

        Returns:
            The new table
        """
        col1 = self.__get_existing_column(col1)
        col2 = self.__get_existing_column(col2)

        self.verify_columns([col1, col2], {Table.ColumnRequirements.IS_KEY})

        t = Table.create_from_table(self)
        idx = self.displayed_columns.index(col2.name)
        if idx < t.marker:
            t.marker -= 1
        t.set_displayed_columns([c for c in self.displayed_columns if c != col2.name])
        t.derivation = self.derivation.equate(col1, col2)
        t.execute()
        return t

    def swap(self, left: str, right: str):
        t = Table.create_from_table(self)
        i = find_index(left, t.displayed_columns)
        j = find_index(right, t.displayed_columns)
        assert i + 1 == j
        # TODO: check that left not strong key for right
        # TODO: modify derivation tree if both keys
        t.displayed_columns = (
            t.displayed_columns[:i]
            + [t.displayed_columns[j], t.displayed_columns[i]]
            + t.displayed_columns[j + 1 :]
        )
        t.execute()
        return t

    def filter(self, by: existing_column):
        column = self.__get_existing_column(by)
        self.verify_columns([column], {Table.ColumnRequirements.IS_KEY_OR_VAL})
        t = Table.create_from_table(self)
        t.execute([Filter(column.get_domain())])
        return t

    def sort(self, cols: list[str]) -> Table:
        # TODO: Semantic sort
        t = Table.create_from_table(self)
        t.execute([Sort(cols)])
        return t

    def rename(self, column: existing_column, new_name: str) -> Table:
        """
        Renames a column in the table

        Args:
            column: the column to be renamed
            new_name: the new name
        """
        column = self.__get_existing_column(column).name

        ## TODO: Raise warning if new_name already exists in the table
        new_name = get_fresh_name(new_name, self.get_namespace())
        t = Table.create_from_table(self)
        idx = find_index(column, t.displayed_columns)
        if idx < 0:
            raise ColumnsNeedToBeInTableException()
        t.derivation = self.derivation.rename(column, new_name)
        t.displayed_columns = (
            t.displayed_columns[:idx] + [new_name] + t.displayed_columns[idx + 1 :]
        )
        t.execute()
        return t

    def __repr__(self):
        left, _, right = self.get_columns_as_lists()
        left = " ".join([l.get_name() for l in left])
        right = " ".join([r.get_name() for r in right])
        dropped_keys = f"\n{self.populated_table.get_num_dropped_keys()} keys hidden"
        dropped_vals = f"\n{self.populated_table.get_num_dropped_vals()} values hidden"
        repr = (
            f"[{left} || {right}]"
            + "\n"
            + str(self.populated_table.get_table_to_display())
        )
        if self.populated_table.get_num_dropped_keys() > 0:
            repr += dropped_keys
        if self.populated_table.get_num_dropped_vals() > 0:
            repr += dropped_vals
        return repr + "\n\n"

    def __str__(self):
        return self.__repr__()

    def __getitem__(self, item):
        col = self.derivation.find_column_with_name(item)
        pt = self.populated_table.copy()
        keys = col.get_strong_keys()
        nodes = [self.derivation.find_node_with_domains([k]) for k in keys]

        if col.is_key_column():
            left = [col]
            right = []
        else:
            left = nodes
            right = [col]

        pt.display(left, right, self.schema.backend)
        left = " ".join([l.get_name() for l in left])
        right = " ".join([r.get_name() for r in right])
        dropped_keys = f"\n{pt.get_num_dropped_keys()} keys hidden"
        dropped_vals = f"\n{pt.get_num_dropped_vals()} values hidden"
        repr = (
                f"[{left} || {right}]"
                + "\n"
                + str(pt.get_table_to_display())
        )
        if pt.get_num_dropped_keys() > 0:
            repr += dropped_keys
        if pt.get_num_dropped_vals() > 0:
            repr += dropped_vals

        if col is not None:
            return Column(col, repr)
        raise KeyError
