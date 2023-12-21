from schema import Cardinality
from tables.raw_column import RawColumn


def find_cardinality_from_new_key_to_hidden_key(key: RawColumn) -> Cardinality:
    keys = key.get_hidden_keys()
    if len(keys) > 1:
        return Cardinality.MANY_TO_MANY
    else:
        return Cardinality.ONE_TO_MANY