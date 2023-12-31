import abc

class Exp(abc.ABC):
    pass

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

