import operator
import typing
from functools import reduce

import numpy as np
import pandas as pd

from schema.node import SchemaNode
from schema.edge import SchemaEdge
from representation.representation import *


class StackPointer:
    def __init__(self, idx, prev=None):
        self.idx = idx
        self.prev = prev

    def __repr__(self):
        return f"SP<{self.idx}>"

    def __str__(self):
        return self.__repr__()


interp = tuple[list, StackPointer]


def cartesian_product(df1: pd.DataFrame, df2: pd.DataFrame) -> pd.DataFrame:
    return df1.merge(df2, how="cross").drop_duplicates()


def get(derivation_step: Get, backend, stack, sp) -> interp:
    columns = derivation_step.columns
    nodes = [c.node for c in columns]
    names = [c.name for c in columns]
    if len(nodes) == 0:
        df = pd.DataFrame()
    elif len(nodes) == 1:
        df = backend.get_domain_from_atomic_node(nodes[0], names[0])
    else:
        domains = [
            backend.get_domain_from_atomic_node(node, name)
            for node, name in zip(nodes, names)
        ]
        df = reduce(cartesian_product, domains)
    return stack + [df], sp


def stt(derivation_step: StartTraversal, backend, stack, sp) -> interp:
    table = stack[-1]
    base = table.copy()
    first_cols = [c.name for c in derivation_step.start_columns]
    df: pd.DataFrame = base[first_cols]
    for i, col in enumerate(first_cols):
        df[i] = df[col]
    return stack + [df], sp


def trv(derivation_step: Traverse, backend, stack, sp) -> interp:
    table = stack[-1]

    start_nodes = derivation_step.edge.from_nodes
    end_nodes = derivation_step.edge.to_nodes

    relation = backend.get_relation_from_mapping(derivation_step.edge, table)

    to_join = list(range(len(start_nodes)))

    df = (
        pd.merge(table, relation, on=to_join, how="right")
        .drop(columns=to_join, axis=1)
        .drop_duplicates()
    )
    df = df.rename(
        {
            k: k - len(start_nodes)
            for k in range(len(start_nodes), len(start_nodes) + len(end_nodes))
        },
        axis=1,
    )

    return stack[:-1] + [df], sp


def prj(derivation_step: Project, _, stack, sp) -> interp:
    table = stack[-1]
    indices = derivation_step.indices
    if len(indices) == 0:
        return stack[:-1] + [pd.DataFrame()], sp
    start_node = derivation_step.start_node
    end_node = derivation_step.end_node
    start_nodes = SchemaNode.get_constituents(start_node)
    end_nodes = SchemaNode.get_constituents(end_node)
    i = 0
    j = 0
    df = table.copy()
    renaming = {j: i for i, j in enumerate(indices)}
    while j < len(end_nodes):
        if indices[j] == i:
            renaming |= {i: j}
            i += 1
            j += 1
        else:
            df = df.drop(i, axis=1)
            i += 1
    df = df.drop(list(range(i, len(start_nodes))), axis=1)
    df = df.rename(renaming, axis=1)
    return stack[:-1] + [df], sp


def exp(derivation_step: Expand, backend, stack, sp) -> interp:
    table = stack[-1]
    end_node = derivation_step.end_node
    end_nodes = SchemaNode.get_constituents(end_node)
    indices = derivation_step.indices

    idxs = [i for i in range(len(end_nodes)) if i not in set(indices)]

    df = table
    exists = set(indices)
    df = df.rename({i: j for i, j in enumerate(indices)}, axis=1)

    for j in range(len(end_nodes)):
        if j not in exists:
            domain = backend.get_domain_from_atomic_node(end_nodes[j], j)
            df = pd.merge(df, domain, how="cross")

    hidden_keys = derivation_step.hidden_keys
    if len(hidden_keys) > 0:
        for i, idx in enumerate(idxs):
            df[hidden_keys[i].name] = df[idx]

    return stack[:-1] + [df], sp


def equ(derivation_step: Equate, _, stack, sp) -> interp:
    return stack, sp


def ent(derivation_step: EndTraversal, _, stack, sp) -> interp:
    cols = [c for c in stack[-1].columns]
    end_cols = [c.name for c in derivation_step.end_columns]
    should_merge = [c not in set(cols) for c in end_cols]
    to_drop = [i for i, b in enumerate(should_merge) if not b]
    renaming = {i: n for (i, n) in enumerate(end_cols) if should_merge[i]}

    x = stack[-1]
    y = stack[-2]
    x = x.drop(to_drop, axis=1).rename(renaming, axis=1)
    common = [col for col in list(x.columns) if col in set(y.columns)]
    if len(common) == 0:
        if len(y.columns) == 0:
            res = x
        elif len(x.columns) == 0:
            res = pd.DataFrame()
        else:
            res = pd.merge(x, y, how="cross")
    else:
        res = pd.merge(x, y, on=common, how="outer")

    res = res.loc[res.astype(str).drop_duplicates().index]

    # TODO: Pass through table
    return stack[:-2] + [res], sp


def rnm(derivation_step: Rename, _, stack, sp) -> interp:
    mapping = derivation_step.mapping
    table = stack[-1]
    return stack[:-1] + [table.rename(mapping, axis=1)], sp


def srt(derivation_step: Sort, _, stack, sp) -> interp:
    table = stack[-1]
    columns = derivation_step.columns
    return stack[:-1] + [table.sort_values(by=columns)], sp


def flt(derivation_step: Filter, _, stack, sp) -> interp:
    table = stack[-1]
    col = derivation_step.column
    df = table[(table[col.name].notnull())]

    return stack[:-1] + [df], sp


def psh(step, _, stack, sp) -> interp:
    return stack + [stack[-1]], sp


def pop(step, _, stack, sp) -> interp:
    return stack[:-1], sp


def cal(step, _, stack, sp) -> interp:
    return stack + [stack[-1]], StackPointer(len(stack) - 1, sp)


def ret(step, _, stack, sp) -> interp:
    return stack[:-2] + [stack[-1]], sp.prev


def rst(step, _, stack, sp) -> interp:
    return stack + [stack[sp.idx]], sp


def mer(step, _, stack, sp) -> interp:
    x = stack[-1]
    y = stack[-2]
    common = [col for col in x.columns if col in set(y.columns)]
    if len(common) == 0:
        if len(x.columns) == 0:
            res = y
        elif len(y.columns) == 0:
            res = x
        else:
            res = pd.merge(x, y, how="cross")
    else:
        res = pd.merge(x, y, on=common, how="outer")
    res = res.loc[res.astype(str).drop_duplicates().index]
    return stack[:-2] + [res], sp


def drp(step: Drop, _, stack, sp) -> interp:
    table: pd.DataFrame = stack[-1]
    to_drop = set([c.name for c in step.columns])
    df = table[[col for col in table if col not in to_drop]].drop_duplicates()
    return stack[:-1] + [df], sp


def step(next_step: RepresentationStep, backend, stack: list, sp) -> interp:
    match next_step.name:
        case "GET":
            next_step = typing.cast(Get, next_step)
            return get(next_step, backend, stack, sp)
        case "PSH":
            next_step = typing.cast(Push, next_step)
            return psh(next_step, backend, stack, sp)
        case "POP":
            next_step = typing.cast(Pop, next_step)
            return pop(next_step, backend, stack, sp)
        case "CAL":
            next_step = typing.cast(Call, next_step)
            return cal(next_step, backend, stack, sp)
        case "RET":
            next_step = typing.cast(Return, next_step)
            return ret(next_step, backend, stack, sp)
        case "RST":
            next_step = typing.cast(Reset, next_step)
            return rst(next_step, backend, stack, sp)
        case "MER":
            next_step = typing.cast(Merge, next_step)
            return mer(next_step, backend, stack, sp)
        case "DRP":
            next_step = typing.cast(Drop, next_step)
            return drp(next_step, backend, stack, sp)
        case "STT":
            next_step = typing.cast(StartTraversal, next_step)
            return stt(next_step, backend, stack, sp)
        case "TRV":
            next_step = typing.cast(Traverse, next_step)
            return trv(next_step, backend, stack, sp)
        case "EQU":
            next_step = typing.cast(Equate, next_step)
            return equ(next_step, backend, stack, sp)
        case "PRJ":
            next_step = typing.cast(Project, next_step)
            return prj(next_step, backend, stack, sp)
        case "EXP":
            next_step = typing.cast(Expand, next_step)
            return exp(next_step, backend, stack, sp)
        case "RNM":
            next_step = typing.cast(Rename, next_step)
            return rnm(next_step, backend, stack, sp)
        case "ENT":
            next_step = typing.cast(EndTraversal, next_step)
            return ent(next_step, backend, stack, sp)
        case "FLT":
            next_step = typing.cast(Filter, next_step)
            return flt(next_step, backend, stack, sp)
        case "SRT":
            next_step = typing.cast(Sort, next_step)
            return srt(next_step, backend, stack, sp)


def interpret(steps: list[RepresentationStep], backend) -> pd.DataFrame:
    stack = []
    sp = None
    for s in steps:
        # print(s)
        stack, sp = step(s, backend, stack, sp)
        # print(stack)
    return stack[0]
