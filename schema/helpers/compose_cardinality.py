from schema.cardinality import Cardinality


def compose_cardinality(c1: Cardinality, c2: Cardinality) -> Cardinality:
    """
    If X->Y has cardinality c1 and Y->Z has cardinality c2, then X->Z has cardinality compose_cardinality(c1, c2)

    Args:
        c1 (Cardinality): The cardinality of the first edge
        c2 (Cardinality): The cardinality of the second edge

    Returns:
        Cardinality: The cardinality of the composed edge
    """
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
