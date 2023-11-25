import abc


class Backend(abc.ABC):
    @abc.abstractmethod
    def map_node_to_domain(self, node, data):
        raise NotImplemented()

    @abc.abstractmethod
    def map_edge_to_relation(self, edge, data):
        raise NotImplemented()

    @abc.abstractmethod
    def get_cardinality(self, edge, start):
        raise NotImplemented()

    @abc.abstractmethod
    def execute_query(self, query):
        raise NotImplemented()

