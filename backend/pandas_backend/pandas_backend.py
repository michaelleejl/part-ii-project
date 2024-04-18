import typing

import pandas as pd

from backend.backend import Backend
from backend.pandas_backend.exp_interpreter import exp_interpreter
from backend.pandas_backend.helpers import (
    copy_data,
    get_cols_of_node,
    determine_cardinality,
)
from backend.pandas_backend.interpreter import interpret
from backend.pandas_backend.pandas_populated_table import PandasPopulatedTable
from backend.pandas_backend.relation import DataRelation
from backend.pandas_backend.transform_interpreter import transform_interpreter
from representation.mapping import Mapping
from schema.edge import SchemaEdge, reverse_cardinality
from schema.node import SchemaNode, AtomicNode, SchemaClass
from representation.representation import RepresentationStep, End
from exp.exp import Exp


def interpret_function(function: Exp):
    return exp_interpreter(function)


def generate_hidden_keys(edge: SchemaEdge, data: pd.DataFrame):
    from_node = edge.from_node
    n = len(SchemaNode.get_constituents(from_node))
    data = data.copy()
    if not edge.is_functional():
        j = 0
        for i in range(n, len(data.columns)):
            data[-j - 1] = data[i]
            j += 1
    return data


class PandasBackend(Backend):

    def __init__(self):
        self.node_data = {}
        self.edge_data = {}
        self.edge_funs = {}
        self.derived_tables = {}
        self.clones = {}

    def map_atomic_node_to_domain(self, node, domain: pd.DataFrame | pd.Series) -> None:
        if isinstance(domain, pd.Series):
            domain = pd.DataFrame(domain)
        cs = SchemaNode.get_constituents(node)
        assert len(cs) == 1
        domain = copy_data(domain).dropna().drop_duplicates()
        self.clones[node] = node
        self.node_data[node] = domain

    def get_domain_size(self, node: SchemaNode):
        cs = SchemaNode.get_constituents(node)
        assert len(cs) == 1
        assert node in self.clones
        lookup = node
        while self.clones[lookup] != lookup:
            lookup = self.clones[node]
        return len(self.node_data[lookup])

    def get_domain_from_atomic_node(self, node: SchemaNode, with_name):
        cs = SchemaNode.get_constituents(node)
        assert len(cs) == 1
        assert node in self.clones
        lookup = node
        while self.clones[lookup] != lookup:
            lookup = self.clones[node]
        copy = copy_data(self.node_data[lookup])
        copy.columns = [with_name]
        return copy

    def clone(self, node: SchemaNode, new_node: SchemaNode):
        self.clones[new_node] = node

    def map_edge_to_data_relation(self, edge: SchemaEdge, relation: pd.DataFrame):
        f_node_c = SchemaNode.get_constituents(edge.from_node)
        t_node_c = SchemaNode.get_constituents(edge.to_node)
        assert len(f_node_c + t_node_c) == len(relation.columns)
        df = copy_data(relation.dropna())
        df.columns = list(range(len(df.columns)))
        self.edge_data[edge] = copy_data(df)

    def map_edge_to_closure(
        self,
        edge,
        function: Exp,
        num_args: int,
        rev_target: SchemaNode = None,
        target_idxs: list[int] = None,
    ):
        if rev_target is None:
            target = edge.from_node
            idxs = list(range(num_args))
        else:
            target = rev_target
            idxs = target_idxs
        forward = SchemaEdge(target, edge.to_node, edge.cardinality)
        rev = SchemaEdge(edge.to_node, target, reverse_cardinality(edge.cardinality))
        fun = interpret_function(function)

        def closure(table):
            df = copy_data(table)
            series = pd.Series(fun(df))
            df[num_args] = series
            data = copy_data(pd.DataFrame(df))
            self.map_edge_to_data_relation(forward, data[idxs + [num_args]])
            self.map_edge_to_data_relation(rev, data[[num_args] + idxs])
            self.map_atomic_node_to_domain(
                edge.to_node, pd.DataFrame(data[num_args]).drop_duplicates()
            )
            return df[list(range(num_args + 1))]

        #
        # elif isinstance(function, AggregationFunction):
        #     fun = interpret_aggregation_function(function)
        #     node_to_drop = function.column.raw_column.node
        #     constituents = SchemaNode.get_constituents(edge.from_node)
        #     modified_start_node = SchemaNode.product([c for c in constituents if c != node_to_drop])
        #     forward = SchemaEdge(modified_start_node, edge.to_node, edge.cardinality)
        #     reverse = SchemaEdge(modified_start_node, edge.to_node, reverse_cardinality(edge.cardinality))
        #
        #     def closure(table, explicit_keys):
        #         df = copy_data(table)
        #         n = len(function.column.get_strong_keys()) + 1
        #         keys = list(range(n - 1))
        #         df = df.merge(fun(df)[keys + [n]], on=keys)[keys + [n-1, n]]
        #         data = copy_data(pd.DataFrame(df)).drop(n-1, axis=1).rename({n: n-1}, axis=1)
        #         data = data.loc[data.astype(str).drop_duplicates().index]
        #         if len(data.columns) > 0:
        #             self.edge_data[forward] = data
        #             self.edge_data[reverse] = data[[n-1] + keys]
        #         self.map_atomic_node_to_domain(edge.to_node, pd.DataFrame(data[n-1]).drop_duplicates())
        #         return df.loc[df.astype(str).drop_duplicates().index]
        #
        # else:
        #     raise Exception()

        self.edge_funs[edge] = closure

    # def get_base_relation(self, edge: SchemaEdge):
    #     rev = SchemaEdge(edge.to_node, edge.from_node)

    def get_relation_from_mapping(self, mapping: Mapping, table) -> pd.DataFrame:
        edge = mapping.edge
        rev = SchemaEdge(edge.to_node, edge.from_node)

        n = len(SchemaNode.get_constituents(edge.from_node))
        m = len(SchemaNode.get_constituents(edge.to_node))

        hks = mapping.hidden_keys
        # function edges
        if edge in self.edge_funs:
            data = self.edge_funs[edge](table)

        elif rev in self.edge_funs:
            data = self.edge_funs[rev](table)
            data = data.rename({j: -j for j in range(n, n + m)}, axis=1)
            data = data.rename({i: i + m for i in range(n)}, axis=1)
            data = data.rename({j: (-j) - n for j in range(-n - m + 1, -n + 1)}, axis=1)

        # data edges
        elif edge in self.edge_data:
            data = copy_data(self.edge_data[edge])

        elif rev in self.edge_data:
            data = copy_data(self.edge_data[rev])
            data = data.rename({j: -j for j in range(n, n + m)}, axis=1)
            data = data.rename({i: i + m for i in range(n)}, axis=1)
            data = data.rename({j: (-j) - n for j in range(-n - m + 1, -n + 1)}, axis=1)
        else:
            assert False

        data = generate_hidden_keys(edge, data)

        data, hks = transform_interpreter(
            data,
            hks,
            mapping.transform,
            lambda n, name: self.get_domain_from_atomic_node(n, name),
        )
        data = data.rename({-i - 1: hk.name for (i, hk) in enumerate(hks)}, axis=1)
        return data

    def extend_domain(self, node: AtomicNode, domain_node: SchemaClass):
        domain = self.get_domain_from_atomic_node(domain_node, domain_node.name)
        domain = copy_data(domain)
        domain.columns = self.node_data[node].columns
        self.node_data[node] = (
            pd.concat([self.node_data[node], domain])
            .drop_duplicates()
            .reset_index(drop=True)
        )

    def get_cardinality(self, edge, start_node):
        mapping = self.edge_data[edge]
        end_node = edge.to_node if start_node == edge.from_node else edge.from_node

        if type(mapping) == DataRelation:
            key_cols = get_cols_of_node(mapping.data, end_node)
            val_cols = get_cols_of_node(mapping.data, start_node)
            return determine_cardinality(mapping.data, key_cols, val_cols)

    def execute_query(
        self, table_id, derived_from, derivation_steps: list[RepresentationStep]
    ) -> tuple[PandasPopulatedTable, Backend]:

        last = typing.cast(End, derivation_steps[-1])

        table = interpret(derivation_steps[:-1], self)
        populated = PandasPopulatedTable(table)
        populated.display(last.left, last.right, self)
        return populated, self
