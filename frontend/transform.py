from __future__ import annotations

from frontend.domain import Domain


class Transform:
    """
    Represents a closure property on edges
    """
    def __init__(self, name):
        self.name = name


class Curry(Transform):
    """
    Represents a currying transformation
    If there is an edge from A * B --> C, then there is an edge from A --> (B --> C)
    """
    def __init__(self, to_curry: int, hidden_key: Domain):
        super().__init__("CUR")
        self.to_curry = to_curry
        self.hidden_key = hidden_key

    def __repr__(self):
        return f"CUR <{self.to_curry}, {self.hidden_key}>"

    def __str__(self):
        return self.__repr__()


class Uncurry(Transform):
    """
    Represents an uncurrying transformation
    If there is an edge from A --> (B --> C), then there is an edge from A * B --> C
    """
    def __init__(self, to_uncurry: int, n: int):
        super().__init__("UNC")
        self.to_uncurry = to_uncurry
        self.n = n

    def __repr__(self):
        return f"UNC <{self.to_uncurry}, {self.n}>"

    def __str__(self):
        return self.__repr__()


class Carry(Transform):
    """
    Represents a carrying transformation
    If there is an edge from A --> B, then there is an edge from A * C --> B * C
    """
    def __init__(self, to_carry: Domain, n: int, m: int):
        super().__init__("CAR")
        self.to_carry = to_carry
        self.n = n
        self.m = m

    def __repr__(self):
        return f"CAR <{self.to_carry}, {self.n}, {self.m}>"

    def __str__(self):
        return self.__repr__()


class Drop(Transform):
    """
    Represents a dropping transformation
    Undoes carrying
    """
    def __init__(self, drop_from: int, drop_to: int):
        super().__init__("DRP")
        self.drop_from = drop_from
        self.drop_to = drop_to

    def __repr__(self):
        return f"DRP <{self.drop_from}, {self.drop_to}>"

    def __str__(self):
        return self.__repr__()


class Invert(Transform):
    """
    Represents an inverting transformation
    If there is an edge from A --> B, then there is an edge from B --> A
    """
    def __init__(self, hidden_keys: list[Domain], n: int, m: int, to_exclude: list[int]):
        super().__init__("INV")
        self.hidden_keys = hidden_keys
        self.n = n
        self.m = m
        self.to_exclude = to_exclude

    def __repr__(self):
        return f"INV <{self.hidden_keys}, {self.n}, {self.m}, {self.to_exclude}>"

    def __str__(self):
        return self.__repr__()
