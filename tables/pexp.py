from abc import ABC

from schema.helpers.find_index import find_index
from tables.exp import Exp


class Pexp(Exp, ABC):
    def __init__(self, code):
        self.code = code
