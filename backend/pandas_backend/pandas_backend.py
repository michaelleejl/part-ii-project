import typing

import numpy as np
import pandas as pd

from backend.backend import Backend
from backend.pandas_backend.helpers import copy_data, get_cols_of_node, determine_cardinality
from backend.pandas_backend.interpreter import get, interpret, end
from backend.pandas_backend.relation import Relation, DataRelation
from schema import SchemaEdge, reverse_cardinality
from schema.node import SchemaNode, AtomicNode, SchemaClass
from tables.aggregation import AggregationFunction
from tables.column import Column
from tables.derivation import DerivationStep, Get, End
from tables.function import Function

def interpret_function(function: Function):
    fun = function.function
    arg = function.arguments

    def interpreted_function(t):
        interpreted_args = []
        num_args = 0
        for a in arg:
            if isinstance(a, Column):
                interpreted_args += [t[num_args]]
                num_args += 1
            else:
                interpreted_args += a
        return fun(interpreted_args), num_args

    return interpreted_function


def interpret_aggregation_function(function: AggregationFunction):
    fun = function.function
    col = function.column
    eks = function.column.get_strong_keys()
    n = len(eks) + 1
    def interpreted_function(t):
        if set(eks) == set(col.raw_column.keyed_by):
            # then you are aggregating a function, i.e. this should not be an element wise aggregation but an aggregation of the whole column
            t2 = copy_data(t)
            t2[n] = t2.drop_duplicates()[n-1].aggregate(fun)
            return t2
        else:
            t2 = copy_data(t)
            to_merge = pd.DataFrame(t.groupby(list(range(n-1)))[n-1].agg(list).apply(fun))[n-1].reset_index().rename({n-1: n}, axis=1)
            t2 = pd.merge(t2, to_merge, on=list(range(n-1)), how="right")
            return t2

    return interpreted_function


class PandasBackend(Backend):

    def __init__(self):
        self.node_data = {}
        self.edge_data = {}
        self.edge_funs = {}
        self.derived_tables = {}
        self.clones = {}

    def map_atomic_node_to_domain(self, node, domain: pd.DataFrame) -> None:
        cs = SchemaNode.get_constituents(node)
        assert len(cs) == 1
        domain = copy_data(domain)
        self.clones[node] = node
        self.node_data[node] = domain

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

    def map_edge_to_data_relation(self, edge, relation: pd.DataFrame):
        rev = SchemaEdge(edge.to_node, edge.from_node)
        assert edge not in self.edge_data
        assert rev not in self.edge_data
        f_node_c = SchemaNode.get_constituents(edge.from_node)
        t_node_c = SchemaNode.get_constituents(edge.to_node)
        assert len(f_node_c + t_node_c) == len(relation.columns)
        df = copy_data(relation)
        df.columns = list(range(len(df.columns)))
        self.edge_data[edge] = copy_data(df)

    def map_edge_to_closure_function(self, edge, function: Function | AggregationFunction):
        rev = SchemaEdge(edge.to_node, edge.from_node, reverse_cardinality(edge.cardinality))
        f_node_c = [str(c) for c in SchemaNode.get_constituents(edge.from_node)]
        t_node_c = [str(c) for c in SchemaNode.get_constituents(edge.to_node)]

        if isinstance(function, Function):
            fun = interpret_function(function)

            def closure(table, keys):
                df = copy_data(table)
                series, num_args = pd.Series(fun(df))
                df[num_args] = series
                # to_merge = returned.apply(lambda x: x[0] if isinstance(x, list) else x).reset_index()
                # df = df.merge(to_merge, on=keys, how="left")
                data = copy_data(pd.DataFrame(df))
                self.map_edge_to_data_relation(rev, data[[num_args] + list(range(num_args))])
                self.map_atomic_node_to_domain(edge.to_node, pd.DataFrame(data[num_args]).drop_duplicates())
                return df[list(range(num_args+1))]

        elif isinstance(function, AggregationFunction):
            fun = interpret_aggregation_function(function)
            node_to_drop = function.column.raw_column.node
            constituents = SchemaNode.get_constituents(edge.from_node)
            modified_start_node = SchemaNode.product([c for c in constituents if c != node_to_drop])
            forward = SchemaEdge(modified_start_node, edge.to_node, edge.cardinality)
            reverse = SchemaEdge(modified_start_node, edge.to_node, reverse_cardinality(edge.cardinality))

            def closure(table, explicit_keys):
                df = copy_data(table)
                n = len(function.column.get_strong_keys()) + 1
                keys = list(range(n - 1))
                df = df.merge(fun(df)[keys + [n]], on=keys)[keys + [n-1, n]]
                data = copy_data(pd.DataFrame(df)).drop(n-1, axis=1).rename({n: n-1}, axis=1)
                data = data.loc[data.astype(str).drop_duplicates().index]
                if len(data.columns) > 0:
                    self.edge_data[forward] = data
                    self.edge_data[reverse] = data[[n-1] + keys]
                self.map_atomic_node_to_domain(edge.to_node, pd.DataFrame(data[n-1]).drop_duplicates())
                return df.loc[df.astype(str).drop_duplicates().index]

        else:
            raise Exception()

        self.edge_funs[edge] = closure

    def get_relation_from_edge(self, edge: SchemaEdge, table, keys) -> pd.DataFrame:
        rev = SchemaEdge(edge.to_node, edge.from_node)
        n = len(SchemaNode.get_constituents(edge.from_node))
        m = len(SchemaNode.get_constituents(edge.to_node))
        if edge in self.edge_funs:
            return self.edge_funs[edge](table, keys)
        elif edge in self.edge_data:
            return copy_data(self.edge_data[edge])
        elif rev in self.edge_funs:
            return self.edge_funs[rev](table, keys).rename({i: i+n for i in range(m)} | {j: j for j in range(n)}, axis=1)
        elif rev in self.edge_data:
            return copy_data(self.edge_data[rev]).rename({i: i+n for i in range(m)} | {j+m: j for j in range(n)}, axis=1)

    def extend_domain(self, node: AtomicNode, domain_node: SchemaClass):
        domain = self.get_domain_from_atomic_node(domain_node, domain_node.name)
        domain = copy_data(domain)
        domain.columns = self.node_data[node].columns
        self.node_data[node] = pd.concat([self.node_data[node], domain]).drop_duplicates().reset_index(drop=True)

    def get_cardinality(self, edge, start_node):
        mapping = self.edge_data[edge]
        end_node = edge.to_node if start_node == edge.from_node else edge.from_node

        if type(mapping) == DataRelation:
            key_cols = get_cols_of_node(mapping.data, end_node)
            val_cols = get_cols_of_node(mapping.data, start_node)
            return determine_cardinality(mapping.data, key_cols, val_cols)

    def execute_query(self, table_id, derived_from, derivation_steps: list[DerivationStep]):
        length = len(derivation_steps)
        assert len(derivation_steps) >= 1
        if derived_from is None or derived_from not in self.derived_tables.keys():
            first = typing.cast(Get, derivation_steps[0])
            tbl = get(first, self)
            start_from = 1
            k = lambda x: x
        else:
            tbl, k, start_from = self.derived_tables[derived_from]

        last = typing.cast(End, derivation_steps[-1])
        derivation_steps = derivation_steps[start_from:-1]

        table, cont = interpret(derivation_steps, self, tbl, k)
        self.derived_tables[table_id] = table, cont, length - 1
        x, y, z = end(last, table, cont)
        return x, y, z, self
