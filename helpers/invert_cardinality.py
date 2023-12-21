from schema.cardinality import Cardinality


def invert_cardinality(value: Cardinality) -> Cardinality:
    match value:
        case Cardinality.ONE_TO_MANY:
            return Cardinality.MANY_TO_ONE
        case Cardinality.MANY_TO_ONE:
            return Cardinality.ONE_TO_MANY
        case _:
            return value
