import abc

import pandas as pd

from backend.pandas_backend.helpers import copy_data, check_columns_match


class Relation(abc.ABC):
    pass


class DataRelation:

    def __init__(self, data: pd.DataFrame):
        super().__init__()
        self.data = copy_data(data)

    def update_relation(self, with_extended):
        extended_transformation = copy_data(with_extended.data)
        old_transformation = self.data
        check_columns_match(old_transformation, extended_transformation)
        return (
            pd.concat([old_transformation, extended_transformation])
            .reset_index()
            .drop_duplicates()
        )


class FunctionRelation:

    def __init__(self, function):
        super().__init__()
        self.function = function
