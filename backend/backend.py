import abc


class Backend(abc.ABC):

    @abc.abstractmethod
    def execute_query(self, table_id, derived_from, query):
        raise NotImplemented()

    def clone(self, node, name):
        raise NotImplemented()

    def extend_domain(self, node, domain):
        raise NotImplemented()

    def map_edge_to_closure(self, edge, function, num_args):
        raise NotImplemented()
