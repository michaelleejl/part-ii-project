import typing

import pandas as pd

from backend.backend import Backend
from backend.pandas_backend.helpers import copy_data, get_cols_of_node, determine_cardinality
from backend.pandas_backend.interpreter import get, interpret, end
from backend.pandas_backend.relation import Relation, DataRelation
from schema import SchemaEdge
from schema.node import SchemaNode
from tables.derivation import DerivationStep, Get, End


class PandasBackend(Backend):

    def __init__(self):
        self.node_data = {}
        self.edge_data = {}
        self.derived_tables = {}
        self.clones = {}

    def map_atomic_node_to_domain(self, node, domain: pd.DataFrame) -> None:
        cs = SchemaNode.get_constituents(node)
        assert len(cs) == 1
        assert node not in self.node_data
        domain = copy_data(domain)
        self.clones[node] = node
        self.node_data[node] = domain

    def get_domain_from_atomic_node(self, node: SchemaNode):
        cs = SchemaNode.get_constituents(node)
        assert len(cs) == 1
        assert node in self.clones
        lookup = node
        while self.clones[node] != node:
            lookup = self.clones[node]
        copy = copy_data(self.node_data[lookup])
        copy.columns = [str(node) for _ in copy.columns]
        return copy

    def clone(self, node: SchemaNode, new_node: SchemaNode):
        self.clones[new_node] = node

    def map_edge_to_relation(self, edge, relation: pd.DataFrame):
        rev = SchemaEdge(edge.to_node, edge.from_node)
        assert edge not in self.edge_data
        assert rev not in self.edge_data
        f_node_c = SchemaNode.get_constituents(edge.from_node)
        t_node_c = SchemaNode.get_constituents(edge.to_node)
        mapping = {f.name: str(f) for f in f_node_c} | {t.name: str(t) for t in t_node_c}
        self.edge_data[edge] = copy_data(relation).rename(mapping, axis=1)

    def get_relation_from_edge(self, edge: SchemaEdge) -> pd.DataFrame:
        rev = SchemaEdge(edge.to_node, edge.from_node)
        if edge in self.edge_data:
            return copy_data(self.edge_data[edge])
        elif rev in self.edge_data:
            return copy_data(self.edge_data[rev])

    def extend_domain(self, node: SchemaNode, domain_node: SchemaNode):
        domain = self.get_domain_from_atomic_node(domain_node)
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
        if derived_from is None:
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
        return end(last, table, cont)
