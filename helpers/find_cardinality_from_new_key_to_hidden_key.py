from schema import Cardinality
from tables.domain import Domain


def find_cardinality_from_new_key_to_hidden_key(key: Domain) -> Cardinality:
    keys = key.get_hidden_keys()
    if len(keys) > 1:
        return Cardinality.MANY_TO_MANY
    else:
        return Cardinality.ONE_TO_MANY