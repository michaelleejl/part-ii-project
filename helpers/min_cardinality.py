from schema.cardinality import Cardinality


def min_cardinality(c1: Cardinality, c2: Cardinality):
    if c1 == Cardinality.ONE_TO_ONE or c2 == Cardinality.ONE_TO_ONE:
        return Cardinality.ONE_TO_ONE
    elif c1 == Cardinality.MANY_TO_ONE or c2 == Cardinality.MANY_TO_ONE:
        return Cardinality.MANY_TO_ONE
    elif c1 == Cardinality.ONE_TO_MANY or c2 == Cardinality.ONE_TO_MANY:
        return Cardinality.ONE_TO_MANY
    else:
        return Cardinality.MANY_TO_MANY
