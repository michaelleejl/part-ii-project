from schema.cardinality import Cardinality


def compose_cardinality(c1: Cardinality, c2: Cardinality):
    match c1:
        case Cardinality.ONE_TO_ONE:
            return c2
        case Cardinality.MANY_TO_ONE:
            match c2:
                case Cardinality.ONE_TO_ONE:
                    return c1
                case Cardinality.ONE_TO_MANY:
                    return Cardinality.MANY_TO_MANY
                case _:
                    return c2
        case Cardinality.ONE_TO_MANY:
            match c2:
                case Cardinality.ONE_TO_ONE:
                    return c1
                case Cardinality.MANY_TO_ONE:
                    return Cardinality.MANY_TO_MANY
                case _:
                    return c2
        case Cardinality.MANY_TO_MANY:
            return c1
