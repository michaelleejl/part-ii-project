from __future__ import annotations

import abc
from abc import ABC

from exp.helpers.convert_key_to_idx_and_update_parameters import (
    convert_key_to_idx_and_update_parameters,
)
from exp.helpers.count_aggregation import count_aggregation
from exp.helpers.count_usage import count_usages
from frontend.domain import Domain
from schema.base_types import BaseType
from exp.exp import Exp
from exp.helpers.wrap_bexp import wrap_bexp


class Bexp(Exp, ABC):
    def __init__(self, code: str):
        super().__init__(code, BaseType.BOOL)

    def __eq__(self, other):
        return EqualityBexp(self, wrap_bexp(other))

    def __and__(self, other):
        return AndBexp(self, wrap_bexp(other))

    def __rand__(self, other):
        return AndBexp(wrap_bexp(other), self)

    def __or__(self, other):
        return OrBexp(self, wrap_bexp(other))

    def __ror__(self, other):
        return OrBexp(wrap_bexp(other), self)

    def __invert__(self):
        return NotBexp(self)

    @abc.abstractmethod
    def to_closure(
        self,
        parameters: list[Domain],
        aggregated_over: dict[int, int],
        usages: dict[int, int],
    ) -> tuple[Bexp, list[Domain], dict[int, int], dict[int, int]]:
        raise NotImplemented()


class ColumnBexp(Bexp):
    def __init__(self, column: Domain | int):
        super().__init__("COL")
        self.column: Domain | int = column

    def __repr__(self):
        return f"COL <{self.column}>"

    def __str__(self):
        return self.__repr__()

    def to_closure(
        self,
        parameters: list[Domain],
        aggregated_over: dict[int, int],
        usages: dict[int, int],
    ) -> tuple[ColumnBexp, list[Domain], dict[int, int], dict[int, int]]:
        idx, parameters = convert_key_to_idx_and_update_parameters(
            self.column, parameters
        )
        usages = count_usages(idx, usages)
        return ColumnBexp(idx), parameters, aggregated_over, usages


class ConstBexp(Bexp):
    def __init__(self, constant: bool):
        super().__init__("CNT")
        self.constant: bool = constant

    def __repr__(self):
        return f"CONST <{self.constant}>"

    def __str__(self):
        return self.__repr__()

    def to_closure(
        self,
        parameters: list[Domain],
        aggregated_over: dict[int, int],
        usages: dict[int, int],
    ) -> tuple[ConstBexp, list[Domain], dict[int, int], dict[int, int]]:
        return self, parameters, aggregated_over, usages


class EqualityBexp(Bexp):
    def __init__(self, lexp: Exp, rexp: Exp):
        super().__init__("EQ")
        self.lexp: Exp = lexp
        self.rexp: Exp = rexp

    def __repr__(self):
        return f"EQ <{self.lexp}, {self.rexp}>"

    def __str__(self):
        return self.__repr__()

    def to_closure(
        self,
        parameters: list[Domain],
        aggregated_over: dict[int, int],
        usages: dict[int, int],
    ) -> tuple[EqualityBexp, list[Domain], dict[int, int], dict[int, int]]:
        new_lexp, lparams, aggregated_over, usages = self.lexp.to_closure(
            parameters, aggregated_over, usages
        )
        new_rexp, rparams, aggregated_over, usages = self.rexp.to_closure(
            lparams, aggregated_over, usages
        )
        return EqualityBexp(new_lexp, new_rexp), rparams, aggregated_over, usages


class NABexp(Bexp):
    def __init__(self, exp: Exp):
        super().__init__("NA")
        self.exp: Exp = exp

    def __repr__(self):
        return f"NA <{self.exp}>"

    def __str__(self):
        return self.__repr__()

    def to_closure(
        self,
        parameters: list[Domain],
        aggregated_over: dict[int, int],
        usages: dict[int, int],
    ) -> tuple[NABexp, list[Domain], dict[int, int], dict[int, int]]:
        new_exp, new_params, aggregated_over, usages = self.exp.to_closure(
            parameters, aggregated_over, usages
        )
        return NABexp(new_exp), new_params, aggregated_over, usages


class NotBexp(Bexp):
    def __init__(self, exp: Bexp):
        super().__init__("NOT")
        self.exp: Bexp = exp

    def to_closure(
        self,
        parameters: list[Domain],
        aggregated_over: dict[int, int],
        usages: dict[int, int],
    ) -> tuple[NotBexp, list[Domain], dict[int, int], dict[int, int]]:
        new_exp, new_params, aggregated_over, usages = self.exp.to_closure(
            parameters, aggregated_over, usages
        )
        return NotBexp(new_exp), new_params, aggregated_over, usages


class LessThanBexp(Bexp):
    def __init__(self, lexp: Exp, rexp: Exp):
        super().__init__("LT")
        self.lexp: Exp = lexp
        self.rexp: Exp = rexp

    def __repr__(self):
        return f"LT <{self.lexp}, {self.rexp}>"

    def __str__(self):
        return self.__repr__()

    def to_closure(
        self,
        parameters: list[Domain],
        aggregated_over: dict[int, int],
        usages: dict[int, int],
    ) -> tuple[LessThanBexp, list[Domain], dict[int, int], dict[int, int]]:
        new_lexp, lparams, aggregated_over, usages = self.lexp.to_closure(
            parameters, aggregated_over, usages
        )
        new_rexp, rparams, aggregated_over, usages = self.rexp.to_closure(
            lparams, aggregated_over, usages
        )
        return LessThanBexp(new_lexp, new_rexp), rparams, aggregated_over, usages


class AndBexp(Bexp):
    def __init__(self, lexp: Bexp, rexp: Bexp):
        super().__init__("AND")
        self.lexp: Bexp = lexp
        self.rexp: Bexp = rexp

    def __repr__(self):
        return f"AND <{self.lexp}, {self.rexp}>"

    def __str__(self):
        return self.__repr__()

    def to_closure(
        self,
        parameters: list[Domain],
        aggregated_over: dict[int, int],
        usages: dict[int, int],
    ) -> tuple[AndBexp, list[Domain], dict[int, int], dict[int, int]]:
        new_lexp, lparams, aggregated_over, usages = self.lexp.to_closure(
            parameters, aggregated_over, usages
        )
        new_rexp, rparams, aggregated_over, usages = self.rexp.to_closure(
            lparams, aggregated_over, usages
        )
        return AndBexp(new_lexp, new_rexp), rparams, aggregated_over, usages


class OrBexp(Bexp):
    def __init__(self, lexp: Bexp, rexp: Bexp):
        super().__init__("OR")
        self.lexp: Bexp = lexp
        self.rexp: Bexp = rexp

    def __repr__(self):
        return f"OR <{self.lexp}, {self.rexp}>"

    def __str__(self):
        return self.__repr__()

    def to_closure(
        self,
        parameters: list[Domain],
        aggregated_over: dict[int, int],
        usages: dict[int, int],
    ) -> tuple[OrBexp, list[Domain], dict[int, int], dict[int, int]]:
        new_lexp, lparams, aggregated_over, usages = self.lexp.to_closure(
            parameters, aggregated_over, usages
        )
        new_rexp, rparams, aggregated_over, usages = self.rexp.to_closure(
            lparams, aggregated_over, usages
        )
        return OrBexp(new_lexp, new_rexp), rparams, aggregated_over, usages


class AnyBexp(Bexp):

    def __init__(
        self,
        keys: list[Domain] | list[int],
        hids: list[Domain] | list[int],
        column: Domain | int,
    ):
        super().__init__("ANY")
        self.keys: list[Domain] | list[int] = keys
        self.hids: list[Domain] | list[int] = hids
        self.column: Domain | int = column

    def __repr__(self):
        return f"ANY <{self.keys}, {self.column}>"

    def to_closure(
        self,
        parameters: list[Domain],
        aggregated_over: dict[int, int],
        usages: dict[int, int],
    ) -> tuple[AnyBexp, list[Domain], dict[int, int], dict[int, int]]:
        key_idxs, parameters = Exp.convert_agg_exp_variables(parameters, self.keys)
        hid_idxs, parameters = Exp.convert_agg_exp_variables(parameters, self.hids)

        idx, parameters = convert_key_to_idx_and_update_parameters(
            self.column, parameters
        )

        for i in key_idxs + hid_idxs + [idx]:
            usages = count_usages(i, usages)
        for j in hid_idxs + [idx]:
            aggregated_over = count_aggregation(j, aggregated_over)

        return AnyBexp(key_idxs, hid_idxs, idx), parameters, aggregated_over, usages


class AllBexp(Bexp):

    def __init__(
        self,
        keys: list[Domain] | list[int],
        hids: list[Domain] | list[int],
        column: Domain | int,
    ):
        super().__init__("ALL")
        self.keys: list[Domain] | list[int] = keys
        self.hids: list[Domain] | list[int] = hids
        self.column: Domain | int = column

    def __repr__(self):
        return f"ALL <{self.keys}, {self.column}>"

    def to_closure(
        self,
        parameters: list[Domain],
        aggregated_over: dict[int, int],
        usages: dict[int, int],
    ) -> tuple[AllBexp, list[Domain], dict[int, int], dict[int, int]]:
        key_idxs, parameters = Exp.convert_agg_exp_variables(parameters, self.keys)
        hid_idxs, parameters = Exp.convert_agg_exp_variables(parameters, self.hids)

        idx, parameters = convert_key_to_idx_and_update_parameters(
            self.column, parameters
        )

        for i in key_idxs + hid_idxs + [idx]:
            usages = count_usages(i, usages)
        for j in hid_idxs + [idx]:
            aggregated_over = count_aggregation(j, aggregated_over)

        return AllBexp(key_idxs, hid_idxs, idx), parameters, aggregated_over, usages