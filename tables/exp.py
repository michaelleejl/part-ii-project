import abc

class Exp(abc.ABC):
    pass

    @classmethod
    def convert_exp(cls, exp):
        return exp.to_closure([])

