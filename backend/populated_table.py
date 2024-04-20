import abc

from representation.domain import Domain


class PopulatedTable(abc.ABC):

    @abc.abstractmethod
    def display(self, left, right, backend):
        pass

    @abc.abstractmethod
    def get_table_to_display(self):
        pass

    @abc.abstractmethod
    def get_raw_table(self):
        pass

    @abc.abstractmethod
    def get_num_dropped_keys(self):
        pass

    @abc.abstractmethod
    def get_num_dropped_vals(self):
        pass

    @abc.abstractmethod
    def evaluate_exp(self, exp, keys: list[Domain], modified_keys: list[int]):
        pass

    @abc.abstractmethod
    def group_by(self, keys: list[Domain], val: Domain):
        pass

    @abc.abstractmethod
    def copy(self):
        pass
