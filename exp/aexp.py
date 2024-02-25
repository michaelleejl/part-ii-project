from __future__ import annotations
import abc

from exp.helpers.convert_key_to_idx_and_update_parameters import (
    convert_key_to_idx_and_update_parameters,
)
from exp.helpers.count_aggregation import count_aggregation
from exp.helpers.count_usage import count_usages
from frontend.domain import Domain
from schema.base_types import BaseType
from exp.exp import Exp
from exp.helpers.wrap_aexp import wrap_aexp


class Aexp(Exp, abc.ABC):
    def __init__(self, code: str):
        super().__init__(code, BaseType.FLOAT)

    def __add__(self, other):
        return AddAexp(self, wrap_aexp(other))

    def __radd__(self, other):
        return wrap_aexp(other).__add__(self)

    def __sub__(self, other):
        return SubAexp(self, wrap_aexp(other))

    def __rsub__(self, other):
        return wrap_aexp(other).__sub__(self)

    def __mul__(self, other):
        return MulAexp(self, wrap_aexp(other))

    def __rmul__(self, other):
        return wrap_aexp(other).__mul__(self)

    def __truediv__(self, other):
        return DivAexp(self, wrap_aexp(other))

    def __rtruediv__(self, other):
        return wrap_aexp(other).__rtruediv__(self)

    def __neg__(self):
        return NegAexp(self)

    @abc.abstractmethod
    def to_closure(
        self,
        parameters: list[Domain],
        aggregated_over: dict[int, int],
        usages: dict[int, int],
    ) -> tuple[Aexp, list[Domain], dict[int, int], dict[int, int]]:
        raise NotImplemented()


class ColumnAexp(Aexp):
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
    ) -> tuple[ColumnAexp, list[Domain], dict[int, int], dict[int, int]]:
        idx, parameters = convert_key_to_idx_and_update_parameters(
            self.column, parameters
        )
        usages = count_usages(idx, usages)
        return ColumnAexp(idx), parameters, aggregated_over, usages


class ConstAexp(Aexp):
    def __init__(self, constant: int | float):
        super().__init__("CNT")
        self.constant = constant

    def __repr__(self):
        return f"CONST <{self.constant}>"

    def __str__(self):
        return self.__repr__()

    def to_closure(
        self,
        parameters: list[Domain],
        aggregated_over: dict[int, int],
        usages: dict[int, int],
    ) -> tuple[ConstAexp, list[Domain], dict[int, int], dict[int, int]]:
        return self, parameters, aggregated_over, usages


class NegAexp(Aexp):
    def __init__(self, exp: Aexp):
        super().__init__("NEG")
        self.exp: Aexp = exp

    def __repr__(self):
        return f"NEG <{self.exp}>"

    def __str__(self):
        return self.__repr__()

    def to_closure(
        self,
        parameters: list[Domain],
        aggregated_over: dict[int, int],
        usages: dict[int, int],
    ) -> tuple[NegAexp, list[Domain], dict[int, int], dict[int, int]]:
        subexp = self.exp
        new_exp, new_params, aggregated_over, usages = subexp.to_closure(
            parameters, aggregated_over, usages
        )
        return NegAexp(new_exp), new_params, aggregated_over, usages


class AddAexp(Aexp):
    def __init__(self, lexp: Exp, rexp: Exp):
        super().__init__("ADD")
        self.lexp: Exp = lexp
        self.rexp: Exp = rexp

    def __repr__(self):
        return f"ADD <{self.lexp}, {self.rexp}>"

    def __str__(self):
        return self.__repr__()

    def to_closure(
        self,
        parameters: list[Domain],
        aggregated_over: dict[int, int],
        usages: dict[int, int],
    ) -> tuple[AddAexp, list[Domain], dict[int, int], dict[int, int]]:
        new_lexp, lparams, aggregated_over, usages = self.lexp.to_closure(
            parameters, aggregated_over, usages
        )
        new_rexp, rparams, aggregated_over, usages = self.rexp.to_closure(
            lparams, aggregated_over, usages
        )
        return AddAexp(new_lexp, new_rexp), rparams, aggregated_over, usages


class SubAexp(Aexp):
    def __init__(self, lexp: Exp, rexp: Exp):
        super().__init__("SUB")
        self.lexp: Exp = lexp
        self.rexp: Exp = rexp

    def __repr__(self):
        return f"SUB <{self.lexp}, {self.rexp}>"

    def __str__(self):
        return self.__repr__()

    def to_closure(
        self,
        parameters: list[Domain],
        aggregated_over: dict[int, int],
        usages: dict[int, int],
    ) -> tuple[SubAexp, list[Domain], dict[int, int], dict[int, int]]:
        new_lexp, lparams, aggregated_over, usages = self.lexp.to_closure(
            parameters, aggregated_over, usages
        )
        new_rexp, rparams, aggregated_over, usages = self.rexp.to_closure(
            lparams, aggregated_over, usages
        )
        return SubAexp(new_lexp, new_rexp), rparams, aggregated_over, usages


class MulAexp(Aexp):

    def __init__(self, lexp: Aexp, rexp: Aexp):
        super().__init__("MUL")
        self.lexp: Aexp = lexp
        self.rexp: Aexp = rexp

    def __repr__(self):
        return f"MUL <{self.lexp}, {self.rexp}>"

    def __str__(self):
        return self.__repr__()

    def to_closure(
        self,
        parameters: list[Domain],
        aggregated_over: dict[int, int],
        usages: dict[int, int],
    ) -> tuple[MulAexp, list[Domain], dict[int, int], dict[int, int]]:
        new_lexp, lparams, aggregated_over, usages = self.lexp.to_closure(
            parameters, aggregated_over, usages
        )
        new_rexp, rparams, aggregated_over, usages = self.rexp.to_closure(
            lparams, aggregated_over, usages
        )
        return MulAexp(new_lexp, new_rexp), rparams, aggregated_over, usages


class DivAexp(Aexp):

    def __init__(self, lexp: Aexp, rexp: Aexp):
        super().__init__("DIV")
        self.lexp: Aexp = lexp
        self.rexp: Aexp = rexp

    def __repr__(self):
        return f"DIV <{self.lexp}, {self.rexp}>"

    def __str__(self):
        return self.__repr__()

    def to_closure(
        self,
        parameters: list[Domain],
        aggregated_over: dict[int, int],
        usages: dict[int, int],
    ) -> tuple[DivAexp, list[Domain], dict[int, int], dict[int, int]]:
        new_lexp, lparams, aggregated_over, usages = self.lexp.to_closure(
            parameters, aggregated_over, usages
        )
        new_rexp, rparams, aggregated_over, usages = self.rexp.to_closure(
            lparams, aggregated_over, usages
        )
        return DivAexp(new_lexp, new_rexp), rparams, aggregated_over, usages


class SumAexp(Aexp):

    def __init__(
        self,
        keys: list[Domain] | list[int],
        hids: list[Domain] | list[int],
        column: Domain | int,
    ):
        super().__init__("SUM")
        self.keys: list[Domain] | list[int] = keys
        self.hids: list[Domain] | list[int] = hids
        self.column: Domain | int = column

    def __repr__(self):
        return f"SUM <{self.column}>"

    def to_closure(
        self,
        parameters: list[Domain],
        aggregated_over: dict[int, int],
        usages: dict[int, int],
    ) -> tuple[SumAexp, list[Domain], dict[int, int], dict[int, int]]:
        key_idxs, parameters = Exp.convert_agg_exp_variables(parameters, self.keys)
        hid_idxs, parameters = Exp.convert_agg_exp_variables(parameters, self.hids)

        idx, parameters = convert_key_to_idx_and_update_parameters(
            self.column, parameters
        )

        for i in key_idxs + hid_idxs + [idx]:
            usages = count_usages(i, usages)
        for j in hid_idxs + [idx]:
            aggregated_over = count_aggregation(j, aggregated_over)

        return SumAexp(key_idxs, hid_idxs, idx), parameters, aggregated_over, usages


class MaxAexp(Aexp):

    def __init__(self, keys, hids, column):
        super().__init__("MAX")
        self.keys: list[Domain] | list[int] = keys
        self.hids: list[Domain] | list[int] = hids
        self.column: Domain | int = column

    def __repr__(self):
        return f"MAX <{self.keys}, {self.column}>"

    def to_closure(
        self,
        parameters: list[Domain],
        aggregated_over: dict[int, int],
        usages: dict[int, int],
    ) -> tuple[MaxAexp, list[Domain], dict[int, int], dict[int, int]]:
        key_idxs, parameters = Exp.convert_agg_exp_variables(parameters, self.keys)
        hid_idxs, parameters = Exp.convert_agg_exp_variables(parameters, self.hids)

        idx, parameters = convert_key_to_idx_and_update_parameters(
            self.column, parameters
        )

        for i in key_idxs + [idx]:
            usages = count_usages(i, usages)
        for j in hid_idxs:
            aggregated_over = count_aggregation(j, aggregated_over)

        aggregated_over = count_aggregation(idx, aggregated_over)

        return MaxAexp(key_idxs, hid_idxs, idx), parameters, aggregated_over, usages


class MinAexp(Aexp):

    def __init__(self, keys, hids, column):
        super().__init__("MIN")
        self.keys: list[Domain] | list[int] = keys
        self.hids: list[Domain] | list[int] = hids
        self.column: Domain | int = column

    def __repr__(self):
        return f"MIN <{self.keys}, {self.column}>"

    def to_closure(
        self,
        parameters: list[Domain],
        aggregated_over: dict[int, int],
        usages: dict[int, int],
    ) -> tuple[MinAexp, list[Domain], dict[int, int], dict[int, int]]:
        key_idxs, parameters = Exp.convert_agg_exp_variables(parameters, self.keys)
        hid_idxs, parameters = Exp.convert_agg_exp_variables(parameters, self.hids)

        idx, parameters = convert_key_to_idx_and_update_parameters(
            self.column, parameters
        )

        for i in key_idxs + hid_idxs + [idx]:
            usages = count_usages(i, usages)
        for j in hid_idxs + [idx]:
            aggregated_over = count_aggregation(j, aggregated_over)

        aggregated_over = count_aggregation(idx, aggregated_over)

        return MinAexp(key_idxs, hid_idxs, idx), parameters, aggregated_over, usages