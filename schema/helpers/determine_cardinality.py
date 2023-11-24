from schema.cardinality import Cardinality


def determine_cardinality(df, keys, values) -> Cardinality:
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
