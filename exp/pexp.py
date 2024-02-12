from abc import ABC

from exp.exp import Exp


class Pexp(Exp, ABC):
    def __init__(self, code):
        self.code = code
