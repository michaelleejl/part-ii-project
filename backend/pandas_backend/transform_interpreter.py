import pandas as pd

from frontend.transform import *


def step(transform: Transform, get_fn):
    match transform.name:
        case "CUR":
            cur = transform
            assert isinstance(cur, Curry)
            idx: int = cur.to_curry
            hidden_key: Domain = cur.hidden_key

            def curry(t: pd.DataFrame, hks: list[Domain]):
                t_new = t.copy()
                tot = len(t_new.columns) - len(hks)
                t_new = t_new[[c for c in t_new.columns if c != idx]]
                t_new = t_new.rename({i: i - 1 for i in range(idx + 1, tot)}, axis=1)
                t_new = t_new.rename(
                    {i: i - 1 for i in range(-1, -len(hks) - 1, -1)}, axis=1
                )
                t_new[-1] = t[idx]
                t_new = t_new.reindex(sorted(t_new.columns), axis=1)

                return t_new, [hidden_key] + hks

            return curry

        case "UNC":
            unc = transform
            assert isinstance(unc, Uncurry)
            org: int = unc.to_uncurry
            idx = -org - 1
            n: int = unc.n

            def uncurry(t: pd.DataFrame, hks: list[Domain]):
                t_new = t.copy()
                t_new = t_new[[c for c in t_new.columns if c != idx]]
                tot = len(t.columns) - len(hks)
                t_new = t_new.rename({i: i + 1 for i in range(n, tot)}, axis=1)
                t_new = t_new.rename(
                    {j: j + 1 for j in range(idx, len(hks) - 2, -1)}, axis=1
                )
                t_new[n] = t[idx]
                t_new = t_new.reindex(sorted(t_new.columns), axis=1)
                return t_new, [hk for (i, hk) in enumerate(hks) if i != org]

            return uncurry

        case "CAR":
            car = transform
            assert isinstance(car, Carry)
            to_get = car.to_carry
            n = car.n
            m = car.m
            domain = get_fn(to_get, n)

            def carry(t: pd.DataFrame, hks: list[Domain]):
                t_new = t.copy()
                t_new = t_new.rename({i: i + 1 for i in range(n, n + m)}, axis=1)
                t_new = pd.merge(t_new, domain, how="cross")
                t_new[n + m + 1] = t_new[n]
                t_new = t_new.reindex(sorted(t_new.columns), axis=1)
                return t_new, hks

            return carry

        case "DRP":
            drp = transform
            assert isinstance(drp, Drop)
            drop_from = drp.drop_from
            drop_to = drp.drop_to

            def drop(t: pd.DataFrame, hks: list[Domain]):
                t_new = t.copy()
                tot = len(t_new.columns) - len(hks)
                t_new = t_new[
                    [c for c in t_new.columns if c not in {drop_to, drop_from}]
                ]
                t_new = t_new.rename(
                    {i: i - 1 for i in range(drop_to + 1, drop_from)}, axis=1
                )
                t_new = t_new.rename(
                    {i: i - 2 for i in range(drop_from + 1, tot)}, axis=1
                )
                t_new = t_new.drop_duplicates()
                t_new = t_new.reindex(sorted(t_new.columns), axis=1).reset_index()
                return t_new, hks

            return drop

        case "INV":
            inv = transform
            assert isinstance(inv, Invert)
            n = inv.n
            m = inv.m
            new_hks = inv.hidden_keys
            to_exclude = inv.to_exclude

            def invert(t: pd.DataFrame, _: list[Domain]):
                t_new = t.copy()
                t_new = t_new[[c for c in t.columns if c >= 0]].drop_duplicates()
                t_new = t_new.rename({j: -j for j in range(n, n + m)}, axis=1)
                t_new = t_new.rename({i: i + m for i in range(n)}, axis=1)
                t_new = t_new.rename(
                    {j: (-j) - n for j in range(-n - m + 1, -n + 1)}, axis=1
                )
                if len(new_hks) > 0:
                    j = 0
                    for i in range(m, n + m):
                        if i - m in set(to_exclude):
                            continue
                        t_new[-j - 1] = t_new[i]
                        j += 1
                t_new = t_new.reindex(sorted(t_new.columns), axis=1)
                return t_new, new_hks

            return invert
    return lambda t: t


def steps(transformations: list[Transform], get_fn):
    k = lambda x, y: (x, y)
    for t in transformations:
        k = lambda x, y: step(t, get_fn)(k(x, y))
    return k


def transform_interpreter(
    t: pd.DataFrame, hks: list[Domain], transformations: list[Transform], get_fn
):
    transformation = steps(transformations, get_fn)
    return transformation(t, hks)
