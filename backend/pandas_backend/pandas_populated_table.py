import operator
from functools import reduce

import numpy as np
import pandas as pd

from backend.pandas_backend.exp_interpreter import exp_interpreter
from backend.populated_table import PopulatedTable
from frontend.derivation.derivation_node import ColumnNode
from frontend.domain import Domain
from schema.helpers.find_index import find_index


class PandasPopulatedTable(PopulatedTable):

    def __init__(self, raw_table: pd.DataFrame):
        self.raw_table: pd.DataFrame = raw_table
        self.to_display: pd.DataFrame | None = None
        self.dropped_keys_count: int = 0
        self.dropped_vals_count: int = 0

    def display(self, left: list[ColumnNode], right: list[ColumnNode], backend):
        keys = left
        hidden = [c.get_hidden_keys() for c in keys if c.is_val_column()]
        to_add_set = set()
        to_add = []
        for hid in hidden:
            for h in hid:
                if h not in to_add_set:
                    to_add_set.add(h)
                    to_add += [h]
        keys = keys + to_add
        values = right

        if len(values) == 0:
            keys_count = reduce(
                operator.mul, [backend.get_domain_size(c.get_schema_node()) for c in left]
            )
            self.to_display = pd.DataFrame()
            self.dropped_keys_count = keys_count
            self.dropped_vals_count = 0
        else:
            keys_str = [k.get_name() for k in keys]
            vals_str = [v.get_name() for v in values]
            app = self.raw_table
            app = app.loc[app.astype(str).drop_duplicates().index]
            df = app[keys_str].reset_index(drop=True)
            df = df.loc[df.astype(str).drop_duplicates().index]
            columns_with_hidden_keys_str = []
            columns_with_hidden_keys = []

            for i, val in enumerate(values):
                hidden_dependencies = [
                    v.name for v in val.get_hidden_keys() if v.name not in set(hidden)
                ]

                if len(hidden_dependencies) > 0:
                    to_add = app[keys_str + [val.get_name()]]
                    to_add = to_add.loc[to_add.dropna().index]
                    to_add = to_add.groupby(keys_str)[val.get_name()].agg(list)
                    columns_with_hidden_keys_str += [val.get_name()]
                    columns_with_hidden_keys += [val]
                else:
                    to_add = app[keys_str + [val.get_name()]]
                df = pd.merge(df, to_add, on=keys_str, how="outer")
                df = df.loc[df.astype(str).drop_duplicates().index]

            df[columns_with_hidden_keys_str] = df[columns_with_hidden_keys_str].map(
                lambda d: (
                    d if isinstance(d, list) and not np.all(pd.isnull(np.array(d))) else np.nan
                )
            )
            if len(values) > 0:
                df2 = df.dropna(subset=vals_str, how="all")
            else:
                df2 = df
            df3 = df2.dropna(subset=keys_str, how="any")

            df3[columns_with_hidden_keys_str] = df3[columns_with_hidden_keys_str].map(
                lambda d: (
                    d if isinstance(d, list) and not np.all(pd.isnull(np.array(d))) else []
                )
            )

            df3 = df3.loc[df3.astype(str).drop_duplicates().index].set_index(keys_str)
            keys_count = reduce(
                operator.mul, [backend.get_domain_size(c.get_schema_node()) for c in keys], 1
            )
            dropped_keys_cnt = keys_count - len(df3)
            self.to_display = df3
            self.dropped_keys_count = dropped_keys_cnt
            self.dropped_vals_count = 0

    def get_table_to_display(self):
        assert self.to_display is not None
        return self.to_display

    def get_num_dropped_keys(self):
        return self.dropped_keys_count

    def get_num_dropped_vals(self):
        return self.dropped_vals_count

    def evaluate_exp(self, exp, start: list[Domain], modified_keys: list[int]):
        df = self.raw_table[[k.name for k in start]]
        df = df.rename({k.name: i for i, k in enumerate(start)}, axis=1)
        n = len(start)
        val = pd.DataFrame(exp_interpreter(exp)(df))
        assert len(val.columns) == 1
        val = val.rename(columns={val.columns[0]: n})
        df = df.join(val)
        df = df[modified_keys + [n]]
        return df
