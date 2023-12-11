from schema import Cardinality


class AggregationFunction:
    def __init__(self, function, column):
        self.function = function
        self.column = column
        self.cardinality = Cardinality.MANY_TO_ONE

