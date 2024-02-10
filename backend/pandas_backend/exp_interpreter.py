import typing

import numpy as np
import pandas as pd

from tables.aexp import *
from tables.bexp import *
from tables.exp import *
from tables.sexp import *


def aggregate(t, keys, col, op):
    if len(keys) > 0:
        return pd.merge(
            t[[c for c in t.columns if c != col]],
            t.groupby(keys)[col].agg(list).apply(op).reset_index(),
            on=keys,
        )[col]
    else:
        # TODO
        return col


def exp_interpreter(exp: Exp):
    if isinstance(exp, Aexp):
        return aexp_interpreter(exp)
    elif isinstance(exp, Bexp):
        return bexp_interpreter(exp)
    elif isinstance(exp, Sexp):
        return sexp_interpreter(exp)
    elif isinstance(exp, Exp):
        match exp.code:
            case "POP":
                pop = typing.cast(PopExp, exp)
                keys = pop.keys
                col = pop.column
                return lambda t: aggregate(t, keys, col, lambda x: x[0])
            case "EXT":
                ext = typing.cast(ExtendExp, exp)
                keys = ext.keys
                col = ext.column
                fun = exp_interpreter(ext.fexp)
                return lambda t: t[col].where(t[col].notna(), fun(t))
            case "MSK":
                msk = typing.cast(MaskExp, exp)
                keys = msk.keys
                col = msk.column
                bxp = bexp_interpreter(msk.bexp)

                def anon(t):
                    return t[col].where(bxp(t).notna() & bxp(t), np.nan)

                return anon


def aexp_interpreter(exp: Aexp):
    match exp.code:
        case "COL":
            col = typing.cast(ColumnAexp, exp)
            return lambda t: t[col.column]
        case "CNT":
            cnt = typing.cast(ConstAexp, exp)
            return lambda t: cnt.constant
        case "ADD":
            add = typing.cast(AddAexp, exp)
            lexp = exp_interpreter(add.lexp)
            rexp = exp_interpreter(add.rexp)
            return lambda t: lexp(t) + rexp(t)
        case "SUB":
            sub = typing.cast(SubAexp, exp)
            lexp = exp_interpreter(sub.lexp)
            rexp = exp_interpreter(sub.rexp)
            return lambda t: lexp(t) - rexp(t)
        case "MUL":
            mul = typing.cast(MulAexp, exp)
            lexp = exp_interpreter(mul.lexp)
            rexp = exp_interpreter(mul.rexp)
            return lambda t: lexp(t) * rexp(t)
        case "DIV":
            div = typing.cast(DivAexp, exp)
            lexp = exp_interpreter(div.lexp)
            rexp = exp_interpreter(div.rexp)
            return lambda t: lexp(t) / rexp(t)
        case "NEG":
            neg = typing.cast(NegAexp, exp)
            sexp = exp_interpreter(neg.exp)
            return lambda t: -sexp(t)
        case "SUM":
            som = typing.cast(SumAexp, exp)
            keys = som.keys
            col = som.column
            return lambda t: aggregate(t, keys, col, np.sum)
        case "MAX":
            oll = typing.cast(MaxAexp, exp)
            keys = oll.keys
            col = oll.column
            return lambda t: aggregate(t, keys, col, np.max)
        case "MIN":
            oll = typing.cast(MaxAexp, exp)
            keys = oll.keys
            col = oll.column
            return lambda t: aggregate(t, keys, col, np.min)
        case "COU":
            cou = typing.cast(CountAexp, exp)
            keys = cou.keys
            col = cou.column
            return lambda t: aggregate(t, keys, col, len)


def bexp_interpreter(exp: Bexp):
    match exp.code:
        case "COL":
            col = typing.cast(ColumnBexp, exp)
            return lambda t: t[col.column]
        case "CNT":
            cnt = typing.cast(ConstBexp, exp)
            return lambda t: cnt.constant
        case "EQ":
            eq = typing.cast(EqualityBexp, exp)
            lexp = exp_interpreter(eq.lexp)
            rexp = exp_interpreter(eq.rexp)
            return lambda t: lexp(t) == rexp(t)
        case "LT":
            lt = typing.cast(LessThanBexp, exp)
            lexp = exp_interpreter(lt.lexp)
            rexp = exp_interpreter(lt.rexp)
            return lambda t: lexp(t) < rexp(t)
        case "NA":
            na = typing.cast(NABexp, exp)
            sexp = exp_interpreter(na.exp)
            return lambda t: sexp(t).isnull()
        case "NOT":
            nt = typing.cast(NotBexp, exp)
            sexp = exp_interpreter(nt.exp)
            return lambda t: ~sexp(t)
        case "AND":
            an = typing.cast(AndBexp, exp)
            lexp = exp_interpreter(an.lexp)
            rexp = exp_interpreter(an.rexp)
            return lambda t: lexp(t) & rexp(t)
        case "OR":
            rr = typing.cast(AndBexp, exp)
            lexp = exp_interpreter(rr.lexp)
            rexp = exp_interpreter(rr.rexp)
            return lambda t: lexp(t) | rexp(t)
        case "ANY":
            ani = typing.cast(AnyBexp, exp)
            keys = ani.keys
            col = ani.column
            return lambda t: aggregate(t, keys, col, np.any)
        case "ALL":
            oll = typing.cast(AllBexp, exp)
            keys = oll.keys
            col = oll.column
            return lambda t: aggregate(t, keys, col, np.all)


def sexp_interpreter(exp: Sexp):
    match exp.code:
        case "COL":
            col = typing.cast(ColumnSexp, exp)
            return lambda t: t[col.column]
        case "CNT":
            cnt = typing.cast(ConstSexp, exp)
            return lambda t: cnt.constant
