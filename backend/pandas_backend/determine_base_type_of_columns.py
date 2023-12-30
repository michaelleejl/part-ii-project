import numpy as np
import pandas as pd
from pandas.api.types import is_string_dtype, is_numeric_dtype, is_bool_dtype

from schema.base_types import BaseType


def convert_dtype_to_base_type(dataframe: pd.DataFrame, column_name: str) -> BaseType:
    column = dataframe[column_name]
    if is_string_dtype(column):
        return BaseType.STRING
    if is_numeric_dtype(column):
        return BaseType.FLOAT
    if is_bool_dtype(column):
        return BaseType.BOOL
    return BaseType.OBJECT

def determine_base_type_of_columns(dataframe: pd.DataFrame) -> list[BaseType]:
    return [convert_dtype_to_base_type(dataframe, c) for c in dataframe.columns]
