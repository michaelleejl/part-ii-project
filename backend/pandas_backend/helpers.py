import pandas as pd

from backend.pandas_backend.exceptions import UpdatingDataShouldPreserveColumnsException
from schema import Cardinality, SchemaNode


def determine_cardinality(df: pd.DataFrame, keys, values) -> Cardinality:
    right = len(df[values].drop_duplicates()) == len(df[values])
    left = len(df[keys].drop_duplicates()) == len(df[keys])
    if left and right:
        return Cardinality.ONE_TO_ONE
    elif left:
        return Cardinality.MANY_TO_ONE
    elif right:
        return Cardinality.ONE_TO_MANY
    else:
        return Cardinality.MANY_TO_MANY


def copy_data(data: pd.DataFrame) -> pd.DataFrame:
    return data.copy()


def get_cols_of_node(df: pd.DataFrame, node: SchemaNode) -> list[str]:
    return list(filter(lambda c: c.startswith(str(hash(node))), df.columns))


def check_columns_match(old_data: pd.DataFrame, new_data: pd.DataFrame) -> None:
    old_columns = set(old_data.columns)
    new_columns = set(new_data.columns)
    additional_columns = new_columns.difference(old_columns)
    missing_columns = old_columns.difference(new_columns)
    if len(additional_columns) > 0 or len(missing_columns) > 0:
        raise UpdatingDataShouldPreserveColumnsException(additional_columns, missing_columns)