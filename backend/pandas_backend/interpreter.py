import itertools
import operator
import typing
from collections import deque
from functools import reduce

import numpy as np
import pandas as pd

from backend.pandas_backend.exp_interpreter import bexp_interpreter
from schema import SchemaNode, SchemaEdge
from tables.column import Column
from tables.derivation import DerivationStep, Get, End, StartTraversal, Traverse, Equate, Project, EndTraversal, Rename, \
    Expand, Filter, Sort
from tables.bexp import *

def cartesian_product(df1: pd.DataFrame, df2: pd.DataFrame) -> pd.DataFrame:
    return df1.merge(df2, how="cross").drop_duplicates()


def get(derivation_step: Get, backend) -> pd.DataFrame:
    columns = derivation_step.columns
    nodes = [c.node for c in columns]
    names = [c.name for c in columns]
    if len(nodes) == 1:
        df = backend.get_domain_from_atomic_node(nodes[0], names[0])
    else:
        domains = [backend.get_domain_from_atomic_node(node, name) for node, name in zip(nodes, names)]
        df = reduce(cartesian_product, domains)
    return df


def end(derivation_step: End, backend, table: pd.DataFrame) -> tuple[pd.DataFrame, int, int]:
    keys = derivation_step.keys
    values = derivation_step.values
    if len(values) == 0:
        keys_count = reduce(operator.mul, [backend.get_domain_size(c.node) for c in keys])
        return pd.DataFrame(), keys_count, 0
    keys_str = [str(k) for k in keys]
    vals_str = [str(v) for v in values]
    app = table
    app = app.loc[app.astype(str).drop_duplicates().index]
    for k in keys_str:
        app[f"KEY_{k}"] = app[k]
    keys_str_with_marker = [f"KEY_{k}" for k in keys_str]
    df = app[keys_str_with_marker].reset_index(drop=True)
    columns_with_hidden_keys_str = []
    columns_with_hidden_keys = []

    for i, val in enumerate(values):
        hidden_dependencies = [str(v) for v in val.get_hidden_keys()]

        if len(hidden_dependencies) > 0:
            to_add = app[keys_str_with_marker + [str(val)]].groupby(keys_str_with_marker)[str(val)].agg(list)
            columns_with_hidden_keys_str += [str(val)]
            columns_with_hidden_keys += [val]
        else:
            to_add = app[keys_str_with_marker + [str(val)]]
        df = pd.merge(df, to_add, on=keys_str_with_marker, how="outer")
        df = df.loc[df.astype(str).drop_duplicates().index]

    df[columns_with_hidden_keys_str] = df[columns_with_hidden_keys_str].map(
        lambda d: d if isinstance(d, list) and not np.all(pd.isnull(np.array(d))) else np.nan)
    if len(values) > 0:
        df2 = df.dropna(subset=vals_str, how="all")
    else:
        df2 = df
    df3 = df2.dropna(subset=keys_str_with_marker, how="any")

    df3[columns_with_hidden_keys_str] = df3[columns_with_hidden_keys_str].map(lambda d: d if isinstance(d, list) and not np.all(pd.isnull(np.array(d))) else [])

    df3 = df3.loc[df3.astype(str).drop_duplicates().index].set_index(keys_str_with_marker)
    renaming = keys_str
    df3.index.set_names(renaming, inplace=True)
    keys_count = reduce(operator.mul, [backend.get_domain_size(c.node) for c in keys])
    dropped_keys_cnt = keys_count - len(df3)
    return df3, dropped_keys_cnt, 0


def stt(derivation_step: StartTraversal, backend, table, stack, keys) -> tuple[pd.DataFrame, list, list]:
    base = table.copy()
    keys = derivation_step.explicit_keys
    first_cols = [c.name for c in derivation_step.start_columns]
    df: pd.DataFrame = base[first_cols]
    for i, col in enumerate(first_cols):
        df[i] = df[col]
    return df, [base], keys


def trv(derivation_step: Traverse, backend, table, stack, keys) -> tuple[pd.DataFrame, list, list]:
    start_node = derivation_step.start_node
    start_nodes = SchemaNode.get_constituents(start_node)
    end_node = derivation_step.end_node
    end_nodes = SchemaNode.get_constituents(end_node)

    relation = backend.get_relation_from_edge(SchemaEdge(start_node, end_node), table, keys)

    hidden_keys = [c for c in derivation_step.hidden_keys]
    idxs = []
    i = 0
    for hk in hidden_keys:
        while end_nodes[i] != hk:
            i += 1
        idxs += [i]
    new_cols = derivation_step.columns

    to_join = list(range(len(start_nodes)))

    df = pd.merge(table, relation, on=to_join, how="right").drop(columns=to_join, axis=1).drop_duplicates()
    df = df.rename({k: k-len(start_nodes) for k in range(len(start_nodes), len(start_nodes) + len(end_nodes))}, axis=1)
    for i, idx in enumerate(idxs):
        df[new_cols[i].name] = df[idx]

    return df, stack, keys + [c.name for c in new_cols]


def prj(derivation_step: Project, _, table, stack, keys) -> tuple[pd.DataFrame, list, list]:
    start_node = derivation_step.start_node
    end_node = derivation_step.end_node
    start_nodes = SchemaNode.get_constituents(start_node)
    end_nodes = SchemaNode.get_constituents(end_node)
    hidden_keys = derivation_step.hidden_keys
    indices = derivation_step.indices
    columns = derivation_step.columns
    i = 0
    j = 0
    k = 0
    df = table.copy()
    retained = []
    renaming = {j:i for i, j in enumerate(indices)}
    while j < len(end_nodes):
        if indices[j] == i:
            renaming |= {i: j}
            i += 1
            j += 1
        else:
            if k < len(hidden_keys) and start_nodes[i] == hidden_keys[k]:
                column = columns[k]
                if column not in set(keys):
                    df[column.name] = df[i]
                    retained += [column.name]
                k += 1
            df = df.drop(i, axis=1)
            i += 1
    df = df.drop(list(range(i, len(start_nodes))), axis=1)
    df = df.rename(renaming, axis=1)
    return df, stack, keys + retained


def exp(derivation_step: Expand, backend, table, stack, keys) -> tuple[pd.DataFrame, list, list]:
    start_node = derivation_step.start_node
    end_node = derivation_step.end_node
    start_nodes = SchemaNode.get_constituents(start_node)
    end_nodes = SchemaNode.get_constituents(end_node)
    indices = derivation_step.indices
    hidden_keys = [c for c in derivation_step.hidden_keys]
    idxs = []
    i = 0

    for hk in hidden_keys:
        while end_nodes[i] != hk:
            i += 1
        idxs += [i]

    df = table
    exists = set(indices)
    df = df.rename({i: j for i, j in enumerate(indices)})

    for j in range(len(end_nodes)):
        if j not in exists:
            domain = backend.get_domain_from_atomic_node(end_nodes[j], j)
            df = pd.merge(df, domain, how="cross")

    new_cols = derivation_step.columns
    for i, idx in enumerate(idxs):
        df[new_cols[i].name] = df[idx]

    return df, stack, keys + [c.name for c in new_cols]


def equ(derivation_step: Equate, _, table, stack, keys) -> tuple[pd.DataFrame, list, list]:
    return table, stack, keys


def ent(derivation_step: EndTraversal, _, table, stack, keys) -> tuple[pd.DataFrame, list, list]:
    cols = [c.name for c in derivation_step.start_columns]
    end_cols = [c.name for c in derivation_step.end_columns]
    should_merge = [c not in set(cols) for c in end_cols]
    to_drop = [i for i, b in enumerate(should_merge) if not b]
    renaming = {i: n for (i, n) in enumerate(end_cols) if should_merge[i]}

    df = pd.merge(table.drop(to_drop, axis=1).rename(renaming, axis=1), stack[0], on=list(set(cols)), how="outer")
    df = df.loc[df.astype(str).drop_duplicates().index]

    # TODO: Pass through table
    return df, [], keys


def rnm(derivation_step: Rename, _, table, stack, keys) -> tuple[pd.DataFrame, list, list]:
    mapping = derivation_step.mapping
    return table.rename(mapping, axis=1), stack, keys


def srt(derivation_step: Sort, _, table, stack, keys) -> tuple[pd.DataFrame, list, list]:
    columns = derivation_step.columns
    return table.sort_values(by=columns), stack, keys


def flt(derivation_step: Filter, _, table, stack, keys) -> tuple[pd.DataFrame, list, list]:
    exp = derivation_step.exp
    arguments = derivation_step.arguments
    renaming = {c.name:i for (i, c) in enumerate(arguments)}
    pred = bexp_interpreter(exp)
    df = table[pred(table.rename(renaming, axis=1))]

    return df, stack, keys


def step(next_step: DerivationStep, backend, table: pd.DataFrame, stack: list, keys) -> tuple[pd.DataFrame, list, list]:
    match next_step.name:
        case "STT":
            next_step = typing.cast(StartTraversal, next_step)
            return stt(next_step, backend, table, [], keys)
        case "TRV":
            next_step = typing.cast(Traverse, next_step)
            return trv(next_step, backend, table, stack, keys)
        case "EQU":
            next_step = typing.cast(Equate, next_step)
            return equ(next_step, backend, table, stack, keys)
        case "PRJ":
            next_step = typing.cast(Project, next_step)
            return prj(next_step, backend, table, stack, keys)
        case "EXP":
            next_step = typing.cast(Expand, next_step)
            return exp(next_step, backend, table, stack, keys)
        case "RNM":
            next_step = typing.cast(Rename, next_step)
            return rnm(next_step, backend, table, stack, keys)
        case "ENT":
            next_step = typing.cast(EndTraversal, next_step)
            return ent(next_step, backend, table, stack, keys)
        case "FLT":
            next_step = typing.cast(Filter, next_step)
            return flt(next_step, backend, table, stack, keys)
        case "SRT":
            next_step = typing.cast(Sort, next_step)
            return srt(next_step, backend, table, stack, keys)


def interpret(steps: list[DerivationStep], backend, table: pd.DataFrame) -> pd.DataFrame:
    if len(steps) == 0:
        return table
    tbl = table
    stack = []
    keys = []
    for s in steps:
        tbl, stack, keys = step(s, backend, tbl, stack, keys)
    return tbl
