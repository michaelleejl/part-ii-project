from __future__ import annotations

import abc
from abc import ABC

from exp.helpers.convert_key_to_idx_and_update_parameters import (
    convert_key_to_idx_and_update_parameters,
)
from exp.helpers.count_usage import count_usages
from frontend.derivation.derivation_node import ColumnNode
from frontend.domain import Domain
from schema.base_types import BaseType
from exp.exp import Exp


class Sexp(Exp, ABC):
    def __init__(self, code):
        super().__init__(code, BaseType.STRING)

    @abc.abstractmethod
    def to_closure(
        self,
        parameters: list[Domain],
        aggregated_over: dict[int, int],
        usages: dict[int, int],
    ) -> tuple[Sexp, list[Domain], dict[int, int], dict[int, int]]:
        raise NotImplemented()


class ColumnSexp(Sexp):
    def __init__(self, column):
        super().__init__("COL")
        self.column = column

    def __repr__(self):
        return f"COL <{self.column}>"

    def __str__(self):
        return self.__repr__()

    def to_closure(
        self,
        parameters: list[ColumnNode],
        aggregated_over: dict[int, int],
        usages: dict[int, int],
    ) -> tuple[ColumnSexp, list[ColumnNode], dict[int, int], dict[int, int]]:
        idx, parameters = convert_key_to_idx_and_update_parameters(
            self.column, parameters
        )
        usages = count_usages(idx, usages)
        return ColumnSexp(idx), parameters, aggregated_over, usages


class ConstSexp(Sexp):
    def __init__(self, constant):
        super().__init__("CNT")
        self.constant = constant

    def __repr__(self):
        return f"CNT <{self.constant}>"

    def __str__(self):
        return self.__repr__()

    def to_closure(
        self,
        parameters: list[ColumnNode],
        aggregated_over: dict[int, int],
        usages: dict[int, int],
    ) -> tuple[ConstSexp, list[ColumnNode], dict[int, int], dict[int, int]]:
        return self, parameters, aggregated_over, usages
