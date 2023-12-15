import itertools
import typing
from collections import deque
from functools import reduce

import numpy as np
import pandas as pd

from schema import SchemaNode, SchemaEdge
from tables.column import Column
from tables.derivation import DerivationStep, Get, End, StartTraversal, Traverse, Equate, Project, EndTraversal, Rename, \
    Expand, Filter, Sort
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


def end(derivation_step: End, table: pd.DataFrame, cont) -> tuple[pd.DataFrame, int, int]:
    keys = derivation_step.keys
    values = derivation_step.values
    keys_str = [str(k) for k in keys]
    vals_str = [str(v) for v in values]
    app = cont(table)
    app = app.loc[app.astype(str).drop_duplicates().index]
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
                break
            candidates = deque()
            candidates.append(hd.keyed_by)
            visited = set()
            while len(candidates) > 0:
                u = candidates.popleft()
                if len(set([str(v) for v in u])) > 0 and set([str(v) for v in u]).issubset(set(keys_str + vals_str[:i]) - set(columns_with_hidden_keys_str)):
                    hidden_dependencies.discard(str(hd))

                def replace_with_dependencies(cols):
                    if len(cols) == 0:
                        return []
                    else:
                        xs = replace_with_dependencies(cols[:-1])
                        if len(xs) == 0:
                            return [cols[0].keyed_by]
                        else:
                            return [[cols[0]] + x for x in xs if x != cols[:-1]] + [cols[0].keyed_by + x for x in xs]

                new_cands = replace_with_dependencies(u)

                for nc in new_cands:
                    if tuple(nc) not in visited:
                        visited.add(tuple(nc))
                        candidates.appendleft(nc)

        if len(hidden_dependencies) > 0:
            to_add = app[keys_str_with_marker + [str(val)]].groupby(keys_str_with_marker)[str(val)].agg(list)
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

    df3[columns_with_hidden_keys_str] = df3[columns_with_hidden_keys_str].map(lambda d: d if isinstance(d, list) and not np.all(pd.isnull(np.array(d))) else [])

    df3 = df3.loc[df3.astype(str).drop_duplicates().index].set_index(keys_str_with_marker)
    renaming = keys_str
    df3.index.set_names(renaming, inplace=True)
    dropped_vals_cnt = len(df2) - len(df3)
    dropped_keys_cnt = len(df) - len(df2)
    return df3, dropped_keys_cnt, dropped_vals_cnt


# TODO: Edit
def stt(derivation_step: StartTraversal, backend, table, cont, stack, keys) -> tuple[pd.DataFrame, any, list, list]:
    base = table.copy()
    keys = derivation_step.explicit_keys
    first_cols = [c.name for c in derivation_step.start_columns]
    df = cont(base)[first_cols]
    for i, col in enumerate(first_cols):
        df[i] = df[col]
    next_step = derivation_step.step
    return step(next_step, backend, df, cont, [base], keys)
    # if next_step.name == "PRJ":
    #     return table, cont, stack + [base], keys
    # elif next_step.name == "EXP":
    #     node = next_step.node
    #     cs = SchemaNode.get_constituents(node)
    #     domains = [backend.get_domain_from_atomic_node(c) for c in cs if str(c) not in table.columns]
    #     df = reduce(cartesian_product, domains)
    #     dfx = cartesian_product(table, df).drop_duplicates()
    #     return dfx, cont, stack + [base], keys
    # elif next_step.name == "TRV":
    #     start_node = next_step.start_node
    #     cols = get_columns_from_node(start_node)
    #     end_node = next_step.end_node
    #     end_cols = get_columns_from_node(end_node)
    #     relation = backend.get_relation_from_edge(SchemaEdge(start_node, end_node), table, keys)
    #     mapping = {str(col.node): col.name for col in derivation_step.start_columns}
    #     relation = relation.rename(mapping, axis=1)
    #     hidden_keys = next_step.hidden_keys
    #     intersection = set(keys).intersection(set(relation.columns).intersection(set(table.columns)))
    #     df = pd.merge(table, relation, on=list(set(first_cols)), how="right")
    #     df = df.loc[df.astype(str).drop_duplicates().index]
    #     return df, cont, stack + [base] + [str(k) for k in hidden_keys], keys
    # elif next_step.name == "EQU":
    #     start_node = next_step.start_node
    #     end_node = next_step.end_node
    #     start_cols = first_cols
    #     end_cols = get_columns_from_node(end_node)
    #     for s, e in zip(start_cols, end_cols):
    #         table[e] = table[s]
    #     return table, cont, stack + [base], keys


def trv(derivation_step: Traverse, backend, table, cont, stack, keys) -> tuple[pd.DataFrame, any, list, list]:
    start_node = derivation_step.start_node
    start_nodes = SchemaNode.get_constituents(start_node)
    end_node = derivation_step.end_node
    end_nodes = SchemaNode.get_constituents(end_node)

    relation = backend.get_relation_from_edge(SchemaEdge(start_node, end_node), table, keys)

    hidden_keys = derivation_step.hidden_keys
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

    return df, cont, stack, keys


def equ(derivation_step: Equate, _, table, cont, stack, keys) -> tuple[pd.DataFrame, any, list, list]:
    return table, cont, stack, keys


def prj(derivation_step: Project, _, table, cont, stack, keys) -> tuple[pd.DataFrame, any, list, list]:
    start_node = derivation_step.start_node
    end_node = derivation_step.end_node
    start_nodes = SchemaNode.get_constituents(start_node)
    end_nodes = SchemaNode.get_constituents(end_node)
    i = 0
    j = 0
    df = table.copy()
    renaming = {}
    while j < len(end_nodes):
        if start_nodes[i] == end_nodes[j]:
            renaming |= {i: j}
            i += 1
            j += 1
        else:
            df = df.drop(i, axis=1)
            i += 1
    df = df.rename(renaming, axis=1)
    return df, cont, stack, keys


def exp(derivation_step: Expand, backend, table, cont, stack, keys) -> tuple[pd.DataFrame, any, list, list]:
    start_node = derivation_step.start_node
    end_node = derivation_step.end_node
    start_nodes = SchemaNode.get_constituents(start_node)
    end_nodes = SchemaNode.get_constituents(end_node)

    df = table
    i = 0
    j = 0
    exists = set()
    while j < len(end_nodes):
        if start_nodes[i] == end_nodes[j]:
            df.rename({i: j})
            exists.add(j)
            i += 1
            j += 1
        else:
            j += 1
    j = 0
    while j < len(end_nodes):
        if j not in exists:
            domain = backend.get_domain_from_atomic_node(end_nodes[j], j)
            df = pd.merge(df, domain, how="cross")

    return df, cont, stack, keys


def ent(derivation_step: EndTraversal, _, table, cont, stack, keys) -> tuple[pd.DataFrame, any, list, list]:
    cols = [c.name for c in derivation_step.start_columns]
    end_cols = derivation_step.end_column.name
    renaming = {0: end_cols}

    def kont(x):
        to_merge = cont(x)
        df = pd.merge(table.rename(renaming, axis=1), to_merge, on=list(set(cols)), how="outer")
        df = df.loc[df.astype(str).drop_duplicates().index]
        return df

    # TODO: Pass through table
    return stack[0], kont, [], keys


def rnm(derivation_step: Rename, _, table, cont, stack, keys) -> tuple[pd.DataFrame, any, list, list]:
    mapping = derivation_step.mapping
    return table.rename(mapping, axis=1), cont, stack, keys


def srt(derivation_step: Sort, _, table, cont, stack, keys) -> tuple[pd.DataFrame, any, list, list]:
    columns = derivation_step.columns
    def kont(x):
        t = cont(x)
        return t.sort_values(by=columns)

    return table, kont, stack, keys


def flt(derivation_step: Filter, _, table, cont, stack, keys) -> tuple[pd.DataFrame, any, list, list]:
    predicate = derivation_step.predicate
    def kont(x):
        t = cont(x)
        pred = predicate_interpreter(predicate)
        return t[pred(t)]

    return table, kont, stack, keys


def step(next_step: DerivationStep, backend, table: pd.DataFrame, cont, stack: list, keys) -> tuple[pd.DataFrame, any, list, list]:
    match next_step.name:
        case "STT":
            next_step = typing.cast(StartTraversal, next_step)
            return stt(next_step, backend, table, cont, [], keys)
        case "TRV":
            next_step = typing.cast(Traverse, next_step)
            return trv(next_step, backend, table, cont, stack, keys)
        case "EQU":
            next_step = typing.cast(Equate, next_step)
            return equ(next_step, backend, table, cont, stack, keys)
        case "PRJ":
            next_step = typing.cast(Project, next_step)
            return prj(next_step, backend, table, cont, stack, keys)
        case "EXP":
            next_step = typing.cast(Expand, next_step)
            return exp(next_step, backend, table, cont, stack, keys)
        case "RNM":
            next_step = typing.cast(Rename, next_step)
            return rnm(next_step, backend, table, cont, stack, keys)
        case "ENT":
            next_step = typing.cast(EndTraversal, next_step)
            return ent(next_step, backend, table, cont, stack, keys)
        case "FLT":
            next_step = typing.cast(Filter, next_step)
            return flt(next_step, backend, table, cont, stack, keys)
        case "SRT":
            next_step = typing.cast(Sort, next_step)
            return srt(next_step, backend, table, cont, stack, keys)


def interpret(steps: list[DerivationStep], backend, table: pd.DataFrame, cont) -> tuple[pd.DataFrame, any]:
    if len(steps) == 0:
        return table, cont
    tbl = table
    stack = []
    keys = []
    for s in steps:
        tbl, cont, stack, keys = step(s, backend, tbl, cont, stack, keys)
    return tbl, cont
