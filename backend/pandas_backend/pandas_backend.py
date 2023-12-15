import typing

import numpy as np
import pandas as pd

from backend.backend import Backend
from backend.pandas_backend.helpers import copy_data, get_cols_of_node, determine_cardinality
from backend.pandas_backend.interpreter import get, interpret, end
from backend.pandas_backend.relation import Relation, DataRelation
from schema import SchemaEdge, reverse_cardinality
from schema.node import SchemaNode
from tables.aggregation import AggregationFunction
from tables.column import Column
from tables.derivation import DerivationStep, Get, End
from tables.function import Function


def interpret_function(function: Function):
    fun = function.function
    arg = function.arguments

    def interpreted_function(t):
        interpreted_args = [t[a.raw_column.name]
                            if isinstance(a, Column) else a for a in arg]
        return fun(interpreted_args)

    return interpreted_function


def interpret_aggregation_function(function: AggregationFunction, name):
    fun = function.function
    col = function.column
    eks = function.column.get_explicit_keys()

    def interpreted_function(t):
        if set(eks) == set(col.raw_column.keyed_by):
            # then you are aggregating a function, i.e. this should not be an element wise aggregation but an aggregation of the whole column
            t2 = copy_data(t)
            t2[col.raw_column.name] = t2.drop_duplicates()[col.raw_column.name].aggregate(fun)
            return t2.rename({str(col.raw_column): name}, axis=1)
        else:
            return pd.DataFrame(t.groupby([str(e) for e in eks])[col.raw_column.name].agg(list).apply(fun)).rename({str(col.raw_column): name}, axis=1).reset_index()

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
                df[str(edge.to_node)] = pd.Series(fun(df))
                # to_merge = returned.apply(lambda x: x[0] if isinstance(x, list) else x).reset_index()
                # df = df.merge(to_merge, on=keys, how="left")
                data = copy_data(pd.DataFrame(df))
                self.map_edge_to_data_relation(rev, data[t_node_c + f_node_c])
                self.map_atomic_node_to_domain(edge.to_node, pd.DataFrame(data[str(edge.to_node)]).drop_duplicates())
                return df

        elif isinstance(function, AggregationFunction):
            fun = interpret_aggregation_function(function, str(edge.to_node))
            node_to_drop = function.column.raw_column.node
            constituents = SchemaNode.get_constituents(edge.from_node)
            modified_start_node = SchemaNode.product([c for c in constituents if c != node_to_drop])
            forward = SchemaEdge(modified_start_node, edge.to_node, edge.cardinality)
            reverse = SchemaEdge(modified_start_node, edge.to_node, reverse_cardinality(edge.cardinality))

            def closure(table, explicit_keys):
                df = copy_data(table)
                keys = list(set([str(f) for f in function.column.get_explicit_keys()] + explicit_keys))
                df = df.merge(fun(df)[keys + [str(edge.to_node)]], on=keys)
                data = copy_data(pd.DataFrame(df)).rename(mapping, axis=1)
                data_with_start_column_dropped = data[[c for c in data.columns if c != str(node_to_drop)]]
                data_with_start_column_dropped= data_with_start_column_dropped.loc[data_with_start_column_dropped.astype(str).drop_duplicates().index]
                if len(data_with_start_column_dropped.columns) > 0:
                    self.edge_data[forward] = data_with_start_column_dropped
                    self.edge_data[reverse] = data_with_start_column_dropped
                self.map_atomic_node_to_domain(edge.to_node, pd.DataFrame(data[str(edge.to_node)]).drop_duplicates())
                return df.loc[df.astype(str).drop_duplicates().index]

        else:
            raise Exception()

        self.edge_funs[edge] = closure

    def get_relation_from_edge(self, edge: SchemaEdge, table, keys) -> pd.DataFrame:
        rev = SchemaEdge(edge.to_node, edge.from_node)
        if edge in self.edge_funs:
            return self.edge_funs[edge](table, keys)
        elif edge in self.edge_data:
            return copy_data(self.edge_data[edge])
        elif rev in self.edge_funs:
            return self.edge_funs[rev](table, keys)
        elif rev in self.edge_data:
            return copy_data(self.edge_data[rev])

    def extend_domain(self, node: SchemaNode, domain_node: SchemaNode):
        domain = self.get_domain_from_atomic_node(domain_node, str(domain_node))
        cs = SchemaNode.get_constituents(node)
        assert len(cs) == 1
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
