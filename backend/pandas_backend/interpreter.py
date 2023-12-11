import itertools
from collections import deque
from functools import reduce
import typing

import numpy as np
import pandas as pd

from schema import SchemaNode, SchemaEdge
from tables.column import Column
from tables.derivation import DerivationStep, Get, End, StartTraversal, Traverse, Equate, Project, EndTraversal, Rename, \
    Cross, Expand, Filter
from tables.predicate import *


def predicate_interpreter(predicate: Predicate):
    match predicate.predicate_type:
        case "EQ":
            eq = typing.cast(EqualityPredicate, predicate)
            if isinstance(eq.value, Column):
                return lambda t: t[eq.name] == t[eq.value.raw_column.name]
            return lambda t: t[eq.name] == eq.value
        case "LT":
            lt = typing.cast(LessThanPredicate, predicate)
            if isinstance(lt.value, Column):
                return lambda t: t[eq.name] < t[eq.value.raw_column.name]
            return lambda t: t[lt.name] < lt.value
        case "NA":
            na = typing.cast(NAPredicate, predicate)
            return lambda t: t[na.name].isnull()
        case "NOT":
            nt = typing.cast(NotPredicate, predicate)
            p = predicate_interpreter(nt.predicate1)
            return lambda t: ~p(t)
        case "AND":
            an = typing.cast(AndPredicate, predicate)
            p1 = predicate_interpreter(an.predicate1)
            p2 = predicate_interpreter(an.predicate2)
            return lambda t: (p1(t)) & (p2(t))
        case "OR":
            rr = typing.cast(AndPredicate, predicate)
            p1 = predicate_interpreter(rr.predicate1)
            p2 = predicate_interpreter(rr.predicate2)
            return lambda t: (p1(t)) | (p2(t))


def get_columns_from_node(node) -> list[str]:
    return [str(c) for c in SchemaNode.get_constituents(node)]


def cartesian_product(df1: pd.DataFrame, df2: pd.DataFrame) -> pd.DataFrame:
    return df1.merge(df2, how="cross")


def get(derivation_step: Get, backend) -> pd.DataFrame:
    nodes = derivation_step.nodes
    if len(nodes) == 1:
        df = backend.get_domain_from_atomic_node(nodes[0])
    else:
        domains = [backend.get_domain_from_atomic_node(n) for n in nodes]
        df = reduce(cartesian_product, domains)
    return df


def end(derivation_step: End, table: pd.DataFrame, cont) -> tuple[pd.DataFrame, int, int]:
    keys = derivation_step.keys
    values = derivation_step.values
    keys_str = [str(k) for k in keys]
    vals_str = [str(v) for v in values]
    app = cont(table)
    for k in keys_str:
        app[f"KEY_{k}"] = app[k]
    keys_str_with_marker = [f"KEY_{k}" for k in keys_str]
    df = app[keys_str_with_marker].reset_index(drop=True)
    columns_with_hidden_keys_str = []
    columns_with_hidden_keys = []

    for i, val in enumerate(values):
        dependencies = [str(v) for v in val.keyed_by]
        hidden_dependencies = set(dependencies) - (set(keys_str + vals_str[:i]) - set(columns_with_hidden_keys_str))
        hidden_depends = set(val.keyed_by) - (set(keys + values[:i]) - set(columns_with_hidden_keys))
        for hd in hidden_depends:
            if hd in set(columns_with_hidden_keys):
                pass
            candidates = deque()
            candidates.append(hd.keyed_by)
            visited = set()
            while len(candidates) > 0:
                u = candidates.popleft()
                if len(set([str(v) for v in u])) > 0 and set([str(v) for v in u]).issubset(set(keys_str + vals_str[:i]) - set(columns_with_hidden_keys_str)):
                    hidden_dependencies.discard(str(hd))
                new_cands = list(map(list, itertools.product(*[k.keyed_by for k in u])))
                for nc in new_cands:
                    if tuple(nc) not in visited:
                        visited.add(tuple(nc))
                        candidates.appendleft(nc)

        if len(hidden_dependencies) > 0:
            to_add = app[keys_str_with_marker + [str(val)]].drop_duplicates().groupby(keys_str_with_marker)[
                str(val)].agg(list)
            columns_with_hidden_keys_str += [str(val)]
            columns_with_hidden_keys += [val]
        else:
            to_add = app[keys_str_with_marker + [str(val)]]
        df = pd.merge(df, to_add, on=keys_str_with_marker, how="outer")
        df = df.loc[df.astype(str).drop_duplicates().index]

    df[columns_with_hidden_keys_str] = df[columns_with_hidden_keys_str].map(
        lambda d: d if isinstance(d, list) and not np.any(pd.isnull(np.array(d))) else np.nan)
    if len(values) > 0:
        df2 = df.dropna(subset=vals_str, how="all")
    else:
        df2 = df
    df3 = df2.dropna(subset=keys_str_with_marker, how="any")

    df3[columns_with_hidden_keys_str] = df3[columns_with_hidden_keys_str].map(lambda d: d if isinstance(d, list) and not np.any(pd.isnull(np.array(d))) else [])

    df3 = df3.loc[df3.astype(str).drop_duplicates().index].set_index(keys_str_with_marker)
    renaming = keys_str
    df3.index.set_names(renaming, inplace=True)
    dropped_vals_cnt = len(df2) - len(df3)
    dropped_keys_cnt = len(df) - len(df2)
    return df3, dropped_keys_cnt, dropped_vals_cnt


def stt(derivation_step: StartTraversal, backend, table, cont, stack) -> tuple[pd.DataFrame, any, list]:
    base = table.copy(deep=True)
    first_cols = get_columns_from_node(derivation_step.start_node)
    table = cont(table.copy(deep=True))[first_cols]
    next_step = derivation_step.step
    if next_step.name == "PRJ":
        return table, cont, stack + [base]
    elif next_step.name == "EXP":
        node = next_step.node
        cs = SchemaNode.get_constituents(node)
        domains = [backend.get_domain_from_atomic_node(c) for c in cs if str(c) not in table.columns]
        df = reduce(cartesian_product, domains)
        dfx = cartesian_product(table, df)
        return dfx, cont, stack + [base]
    elif next_step.name == "TRV":
        start_node = next_step.start_node
        cols = get_columns_from_node(start_node)
        end_node = next_step.end_node
        relation = backend.get_relation_from_edge(SchemaEdge(start_node, end_node), table)
        hidden_keys = next_step.hidden_keys
        return pd.merge(table, relation, on=cols, how="right"), cont, stack + [base] + [str(k) for k in hidden_keys]
    elif next_step.name == "EQU":
        start_node = next_step.start_node
        end_node = next_step.end_node
        start_cols = get_columns_from_node(start_node)
        end_cols = get_columns_from_node(end_node)
        for s, e in zip(start_cols, end_cols):
            table[e] = table[s]
        return table, cont, stack + [base]


def trv(derivation_step: Traverse, backend, table, cont, stack) -> tuple[pd.DataFrame, any, list]:
    start_node = derivation_step.start_node
    end_node = derivation_step.end_node
    new_cols = get_columns_from_node(end_node)
    hidden_keys = derivation_step.hidden_keys
    mapping = derivation_step.mapping
    cols = get_columns_from_node(start_node)
    to_join = cols + list(mapping.values())
    relation = backend.get_relation_from_edge(SchemaEdge(start_node, end_node), table).rename(mapping, axis=1)
    for nc in new_cols:
        if nc in mapping.keys():
            relation[nc] = relation[mapping[nc]]
    if len(stack) == 1:
        to_drop = []
        new_stack = stack
        acc = []
    else:
        acc = stack[-1]
        to_drop = [c for c in cols if c not in set(acc + new_cols)]
        new_stack = stack[:-1]
    acc += [str(k) for k in hidden_keys]
    return (pd.merge(table, relation, on=to_join, how="right")
            .drop(columns=to_drop, axis=1), cont, new_stack + [acc])


def equ(derivation_step: Equate, backend, table, cont, stack) -> tuple[pd.DataFrame, any, list]:
    start_node = derivation_step.start_node
    end_node = derivation_step.end_node
    start_cols = get_columns_from_node(start_node)
    end_cols = get_columns_from_node(end_node)
    renaming = {s: e for s, e in zip(start_cols, end_cols)}
    return table.rename(renaming, axis=1), cont, stack


def prj(derivation_step: Project, backend, table, cont, stack) -> tuple[pd.DataFrame, any, list]:
    columns = get_columns_from_node(derivation_step.node)
    hks = []
    if len(stack) > 1:
        hks = stack[-1]
    return table[columns + hks], cont, stack


def exp(derivation_step: Expand, backend, table, cont, stack) -> tuple[pd.DataFrame, any, list]:
    node = derivation_step.node
    cs = SchemaNode.get_constituents(node)
    domains = [backend.get_domain_from_atomic_node(c) for c in cs if str(c) not in table.columns]

    df = reduce(cartesian_product, domains)
    dfx = cartesian_product(table, df)

    return dfx, cont, stack


def ent(derivation_step: EndTraversal, backend, table, cont, stack) -> tuple[pd.DataFrame, any, list]:
    cols = get_columns_from_node(derivation_step.start_node)

    def kont(x):
        return pd.merge(cont(x), table, on=cols, how="outer")

    assert len(stack) == 1 or len(stack) == 2
    if len(stack) == 2:
        acc = [stack[1]]
    else:
        acc = []
    return stack[0], kont, acc


def rnm(derivation_step: Rename, backend, table, cont, stack) -> tuple[pd.DataFrame, any, list]:
    mapping = derivation_step.mapping
    return table.rename(mapping, axis=1), cont, stack


def flt(derivation_step: Filter, backend, table, cont, stack) -> tuple[pd.DataFrame, any, list]:
    predicate = derivation_step.predicate
    def kont(x):
        t = cont(x)
        pred = predicate_interpreter(predicate)
        return t[pred(t)]

    return table, kont, stack


def step(next_step: DerivationStep, backend, table: pd.DataFrame, cont, stack: list) -> tuple[pd.DataFrame, any, list]:
    match next_step.name:
        case "STT":
            next_step = typing.cast(StartTraversal, next_step)
            return stt(next_step, backend, table, cont, [])
        case "TRV":
            next_step = typing.cast(Traverse, next_step)
            return trv(next_step, backend, table, cont, stack)
        case "EQU":
            next_step = typing.cast(Equate, next_step)
            return equ(next_step, backend, table, cont, stack)
        case "PRJ":
            next_step = typing.cast(Project, next_step)
            return prj(next_step, backend, table, cont, stack)
        case "EXP":
            next_step = typing.cast(Expand, next_step)
            return exp(next_step, backend, table, cont, stack)
        case "RNM":
            next_step = typing.cast(Rename, next_step)
            return rnm(next_step, backend, table, cont, stack)
        case "ENT":
            next_step = typing.cast(EndTraversal, next_step)
            return ent(next_step, backend, table, cont, stack)
        case "FLT":
            next_step = typing.cast(Filter, next_step)
            return flt(next_step, backend, table, cont, stack)


def interpret(steps: list[DerivationStep], backend, table: pd.DataFrame, cont) -> tuple[pd.DataFrame, any]:
    if len(steps) == 0:
        return table, cont
    tbl = table
    stack = []
    for s in steps:
        tbl, cont, stack = step(s, backend, tbl, cont, stack)
    return tbl, cont
