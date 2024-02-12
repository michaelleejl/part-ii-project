from __future__ import annotations
import abc

from exp.helpers.convert_key_to_idx_and_update_parameters import (
    convert_key_to_idx_and_update_parameters,
)
from exp.helpers.count_aggregation import count_aggregation
from exp.helpers.count_usage import count_usages
from frontend.domain import Domain
from schema import BaseType
from schema.helpers.find_index import find_index


class Exp(abc.ABC):
    def __init__(self, code: str, exp_type: BaseType):
        self.code: str = code
        self.exp_type: BaseType = exp_type

    @classmethod
    def convert_exp(
        cls, exp: Exp
    ) -> tuple[Exp, list[Domain], dict[int, int], dict[int, int]]:
        return exp.to_closure([], {}, {})

    @classmethod
    def convert_agg_exp_variables(
        cls, parameters: list[Domain], keys: list[Domain]
    ) -> tuple[list[int], list[Domain]]:
        key_idxs = [find_index(key, parameters) for key in keys]
        i = len(parameters)
        key_params = []
        for j, col_idx in enumerate(key_idxs):
            if col_idx == -1:
                key_params += [i]
                parameters += [keys[j]]
                i += 1
            else:
                key_params += [col_idx]
        return key_params, parameters

    @abc.abstractmethod
    def to_closure(
        self,
        parameters: list[Domain],
        aggregated_over: dict[int, int],
        usages: dict[int, int],
    ) -> tuple[Exp, list[Domain], dict[int, int], dict[int, int]]:
        raise NotImplemented()


class CountExp(Exp):
    def __init__(
        self,
        keys: list[Domain] | list[int],
        hids: list[Domain] | list[int],
        column: Domain | int,
        exp_type: BaseType,
    ):
        super().__init__("COU", exp_type)
        self.keys: list[Domain] | list[int] = keys
        self.hids: list[Domain] | list[int] = hids
        self.column: Domain | int = column

    def __repr__(self):
        return f"COU <{self.keys}, {self.column}>"

    def to_closure(
        self,
        parameters: list[Domain],
        aggregated_over: dict[int, int],
        usages: dict[int, int],
    ) -> tuple[CountExp, list, dict, dict]:
        key_idxs, parameters = Exp.convert_agg_exp_variables(parameters, self.keys)
        hid_idxs, parameters = Exp.convert_agg_exp_variables(parameters, self.hids)
        idx, parameters = convert_key_to_idx_and_update_parameters(
            self.column, parameters
        )
        for i in key_idxs + hid_idxs + [idx]:
            usages = count_usages(i, usages)
        for j in hid_idxs + [idx]:
            aggregated_over = count_aggregation(j, aggregated_over)

        return (
            CountExp(key_idxs, hid_idxs, idx, self.exp_type),
            parameters,
            aggregated_over,
            usages,
        )


class PopExp(Exp):

    def __init__(
        self,
            keys: list[Domain] | list[int],
            hids: list[Domain] | list[int],
            column: Domain | int,
            exp_type: BaseType
    ):
        super().__init__("POP", exp_type)
        self.keys: list[Domain] | list[int] = keys
        self.hids: list[Domain] | list[int] = hids
        self.column: Domain | int = column

    def __repr__(self):
        return f"POP <{self.keys}, {self.column}>"

    def to_closure(
        self,
        parameters: list[Domain],
        aggregated_over: dict[int, int],
        usages: dict[int, int],
    ) -> tuple[PopExp, list, dict, dict]:
        key_idxs, parameters = Exp.convert_agg_exp_variables(parameters, self.keys)
        hid_idxs, parameters = Exp.convert_agg_exp_variables(parameters, self.hids)

        idx, parameters = convert_key_to_idx_and_update_parameters(
            self.column, parameters
        )
        for i in key_idxs + hid_idxs + [idx]:
            usages = count_usages(i, usages)
        for j in hid_idxs + [idx]:
            aggregated_over = count_aggregation(j, aggregated_over)

        return (
            PopExp(key_idxs, hid_idxs, idx, self.exp_type),
            parameters,
            aggregated_over,
            usages,
        )


class ExtendExp(Exp):

    def __init__(
        self,
        keys: list[Domain] | list[int],
        column: Domain | int,
        fexp: Exp,
        exp_type: BaseType,
    ):
        super().__init__("EXT", exp_type)
        self.keys: list[Domain] | list[int] = keys
        self.column: Domain | int = column
        self.fexp: Exp = fexp

    def __repr__(self):
        return f"EXT <{self.keys}, {self.column}, {self.fexp}>"

    def to_closure(
        self,
        parameters: list[Domain],
        aggregated_over: dict[int, int],
        usages: dict[int, int],
    ) -> tuple[Exp, list[Domain], dict[int, int], dict[int, int]]:
        key_idxs, parameters = Exp.convert_agg_exp_variables(parameters, self.keys)

        idx, parameters = convert_key_to_idx_and_update_parameters(
            self.column, parameters
        )

        for i in key_idxs + [idx]:
            usages = count_usages(i, usages)

        new_fexp, new_params, aggregated_over, usages = self.fexp.to_closure(
            parameters, aggregated_over, usages
        )

        return (
            ExtendExp(key_idxs, idx, self.fexp, self.exp_type),
            new_params,
            aggregated_over,
            usages,
        )


class MaskExp(Exp):
    from exp.bexp import Bexp

    def __init__(
        self,
        keys: list[Domain] | list[int],
        column: Domain | int,
        bexp: Bexp,
        exp_type: BaseType,
    ):
        from bexp import Bexp

        super().__init__("MSK", exp_type)
        self.keys: list[Domain] | list[int] = keys
        self.column: Domain | int = column
        self.bexp: Bexp = bexp

    def __repr__(self):
        return f"MSK <{self.keys}, {self.column}, {self.bexp}>"

    def to_closure(
        self,
        parameters: list[Domain],
        aggregated_over: dict[int, int],
        usages: dict[int, int],
    ) -> tuple[Exp, list[Domain], dict[int, int], dict[int, int]]:
        key_idxs, parameters = Exp.convert_agg_exp_variables(parameters, self.keys)

        idx, parameters = convert_key_to_idx_and_update_parameters(
            self.column, parameters
        )

        for i in key_idxs + [idx]:
            usages = count_usages(i, usages)

        new_bexp, new_params, aggregated_over, usages = self.bexp.to_closure(
            parameters, aggregated_over, usages
        )
        return (
            MaskExp(key_idxs, idx, new_bexp, self.exp_type),
            new_params,
            aggregated_over,
            usages,
        )
