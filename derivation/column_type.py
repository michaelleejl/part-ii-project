import abc


class ColumnType(abc.ABC):

    @abc.abstractmethod
    def is_key_column(self):
        pass

    @abc.abstractmethod
    def is_val_column(self):
        pass

    @abc.abstractmethod
    def get_strong_keys(self, node):
        pass

    @abc.abstractmethod
    def get_hidden_keys(self, node):
        pass

    @abc.abstractmethod
    def get_derivation(self, node):
        pass


class Key(ColumnType):

    def is_key_column(self):
        return True

    def is_val_column(self):
        return False

    def get_strong_keys(self, node):
        return []

    def get_hidden_keys(self, node):
        return []

    def get_derivation(self, node):
        return [], []


class Val(ColumnType):

    def is_key_column(self):
        return False

    def is_val_column(self):
        return True

    def get_strong_keys(self, node):
        return node.find_strong_keys()

    def get_hidden_keys(self, node):
        return node.get_hidden_keys_for_val()

    def get_derivation(self, node):
        return self.get_strong_keys(node), self.get_hidden_keys(node)
