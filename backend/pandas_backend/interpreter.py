from collections import deque
from functools import reduce
import typing

import pandas as pd

from schema import SchemaNode, SchemaEdge
from tables.derivation import DerivationStep, Get, End, StartTraversal, Traverse, Equate, Project, EndTraversal, Rename


def get_columns_from_node(node) -> list[str]:
    return [str(c) for c in SchemaNode.get_constituents(node)]


def get(derivation_step: Get, backend) -> pd.DataFrame:
    nodes = derivation_step.nodes
    if len(nodes) == 1:
        df = backend.get_domain_from_atomic_node(nodes[0])
    else:
        domains = [backend.get_domain_from_atomic_node(n) for n in nodes]

        def cartesian_product(df1: pd.DataFrame, df2: pd.DataFrame) -> pd.DataFrame:
            return df1.merge(df2, how="cross")

        df = reduce(cartesian_product, domains)
    return df


def end(derivation_step: End, table: pd.DataFrame, cont) -> pd.DataFrame:
    app = cont(table)
    keys = derivation_step.keys
    values = derivation_step.values
    filtered_table = app[keys + values]
    filtered_table = filtered_table.dropna(subset=keys, how="any")
    if len(values) > 0:
        filtered_table = filtered_table.dropna(subset=values, how="all")
    return filtered_table


def stt(derivation_step: StartTraversal, backend, table, cont, base) -> tuple[pd.DataFrame, any]:
    table = table.copy(deep=True)
    next_step = derivation_step.step
    if next_step.name == "PRJ":
        return table, cont
    elif next_step.name == "TRV":
        start_node = next_step.start_node
        cols = get_columns_from_node(start_node)
        end_node = next_step.end_node
        relation = backend.get_relation_from_edge(SchemaEdge(start_node, end_node))
        return pd.merge(table, relation, on=cols, how="right"), cont
    elif next_step.name == "EQU":
        start_node = next_step.start_node
        end_node = next_step.end_node
        start_cols = get_columns_from_node(start_node)
        end_cols = get_columns_from_node(end_node)
        for s, e in zip(start_cols, end_cols):
            table[e] = table[s]
        return table, cont


def trv(derivation_step: Traverse, backend, table, cont, base) -> tuple[pd.DataFrame, any]:
    start_node = derivation_step.start_node
    cols = get_columns_from_node(start_node)
    end_node = derivation_step.end_node
    relation = backend.get_relation_from_edge(SchemaEdge(start_node, end_node))
    return pd.merge(table, relation, on=cols, how="right").drop(columns=cols, axis=1), cont


def equ(derivation_step: Equate, backend, table, cont, base) -> tuple[pd.DataFrame, any]:
    start_node = derivation_step.start_node
    end_node = derivation_step.end_node
    start_cols = get_columns_from_node(start_node)
    end_cols = get_columns_from_node(end_node)
    renaming = {s: e for s, e in zip(start_cols, end_cols)}
    return table.rename(renaming, axis=1), cont


def prj(derivation_step: Project, backend, table, cont, base) -> tuple[pd.DataFrame, any]:
    columns = derivation_step.columns
    return table[columns], cont


def ent(derivation_step: EndTraversal, backend, table, cont, base) -> tuple[pd.DataFrame, any]:
    cols = get_columns_from_node(derivation_step.start_node)

    def kont(x):
        return pd.merge(cont(x), table, on=cols, how="outer")

    return base, kont


def rnm(derivation_step: Rename, backend, table, cont, base) -> tuple[pd.DataFrame, any]:
    mapping = derivation_step.mapping
    return table.rename(mapping, axis=1), cont


def step(next_step: DerivationStep, backend, table: pd.DataFrame, cont, base) -> tuple[pd.DataFrame, any]:
    match next_step.name:
        case "STT":
            next_step = typing.cast(StartTraversal, next_step)
            return stt(next_step, backend, table, cont, base)
        case "TRV":
            next_step = typing.cast(Traverse, next_step)
            return trv(next_step, backend, table, cont, base)
        case "EQU":
            next_step = typing.cast(Equate, next_step)
            return equ(next_step, backend, table, cont, base)
        case "PRJ":
            next_step = typing.cast(Project, next_step)
            return prj(next_step, backend, table, cont, base)
        case "RNM":
            next_step = typing.cast(Rename, next_step)
            return rnm(next_step, backend, table, cont, base)
        case "ENT":
            next_step = typing.cast(EndTraversal, next_step)
            return ent(next_step, backend, table, cont, base)


def interpret(steps: list[DerivationStep], backend, table: pd.DataFrame, cont) -> tuple[pd.DataFrame, any]:
    if len(steps) == 0:
        return table, cont
    tbl = table
    for s in steps:
        tbl, cont = step(s, backend, tbl, cont, table)
    return tbl, cont
