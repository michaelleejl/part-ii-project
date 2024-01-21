import abc

from schema.helpers.find_index import find_index


class Exp(abc.ABC):
    def __init__(self, code, exp_type):
        self.code = code
        self.exp_type = exp_type

    @classmethod
    def convert_exp(cls, exp) -> tuple[any, list, list]:
        return exp.to_closure([], [])

    @classmethod
    def convert_agg_exp_variables(cls, parameters, key_idxs, keys):
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
    def to_closure(self, parameters, aggregated_over):
        raise NotImplemented()


class PopExp(Exp):

    def __init__(self, keys, column, exp_type):
        super().__init__("POP", exp_type)
        self.keys = keys
        self.column = column

    def __repr__(self):
        return f"POP <{self.keys}, {self.column}>"

    def to_closure(self, parameters, aggregated_over):
        key_idxs = [find_index(key, parameters) for key in self.keys]
        idx = find_index(self.column, parameters)
        aggregated_over = aggregated_over + [self.column] + self.column.get_hidden_keys()
        key_params, parameters = Exp.convert_agg_exp_variables(parameters, key_idxs, self.keys)
        if idx == -1:
            return PopExp(key_params, len(parameters), self.exp_type), parameters + [self.column], aggregated_over
        else:
            return PopExp(key_params, idx, self.exp_type), parameters, aggregated_over


class ExtendExp(Exp):

    def __init__(self, keys, column, fexp, exp_type):
        super().__init__("EXT", exp_type)
        self.keys = keys
        self.column = column
        self.fexp = fexp

    def __repr__(self):
        return f"EXT <{self.keys}, {self.column}, {self.fexp}>"

    def to_closure(self, parameters, aggregated_over):
        key_idxs = [find_index(key, parameters) for key in self.keys]
        idx = find_index(self.column, parameters)
        aggregated_over = aggregated_over
        key_params, parameters = Exp.convert_agg_exp_variables(parameters, key_idxs, self.keys)
        if idx == -1:
            return ExtendExp(key_params, len(parameters), self.fexp, self.exp_type), parameters + [self.column], aggregated_over
        else:
            return ExtendExp(key_params, idx, self.fexp, self.exp_type), parameters, aggregated_over