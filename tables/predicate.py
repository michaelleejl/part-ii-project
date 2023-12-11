class Predicate:
    def __init__(self, predicate_type):
        self.predicate_type = predicate_type

    def __and__(self, other):
        return AndPredicate(self, other)

    def __rand__(self, other):
        return AndPredicate(self, other)

    def __or__(self, other):
        return OrPredicate(self, other)

    def __ror__(self, other):
        return OrPredicate(self, other)

    def __invert__(self):
        return NotPredicate(self)


class EqualityPredicate(Predicate):
    def __init__(self, name, value):
        super().__init__("EQ")
        self.name = name
        self.value = value

    def __repr__(self):
        return f"EQ <{self.name}, {self.value}>"

    def __str__(self):
        return self.__repr__()


class NAPredicate(Predicate):
    def __init__(self, name):
        super().__init__("NA")
        self.name = name

    def __repr__(self):
        return f"NA <{self.name}>"

    def __str__(self):
        return self.__repr__()


class LessThanPredicate(Predicate):
    def __init__(self, name, value):
        super().__init__("LT")
        self.name = name
        self.value = value

    def __repr__(self):
        return f"LT <{self.name}, {self.value}>"

    def __str__(self):
        return self.__repr__()


class AndPredicate(Predicate):
    def __init__(self, predicate1, predicate2):
        super().__init__("AND")
        self.predicate1 = predicate1
        self.predicate2 = predicate2


class OrPredicate(Predicate):
    def __init__(self, predicate1, predicate2):
        super().__init__("OR")
        self.predicate1 = predicate1
        self.predicate2 = predicate2


class NotPredicate(Predicate):
    def __init__(self, predicate1):
        super().__init__("NOT")
        self.predicate1 = predicate1
