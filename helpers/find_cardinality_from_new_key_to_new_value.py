from helpers.invert_cardinality import invert_cardinality
from schema import Cardinality
from tables.domain import Domain


def find_cardinality_from_new_key_to_new_value(key: Domain) -> Cardinality:
    keys = key.get_strong_keys()
    if len(keys) > 1:
        return Cardinality.MANY_TO_MANY
    else:
        return invert_cardinality(key.cardinality)