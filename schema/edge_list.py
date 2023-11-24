from schema.edge import SchemaEdge
from schema.exceptions import EdgeAlreadyExistsException, EdgeDoesNotExistException


class SchemaEdgeList:
    def __init__(self, edge_list: frozenset[SchemaEdge] = frozenset()):
        self.edge_list = edge_list

    @classmethod
    def add_edge(cls, edge_list, edge: SchemaEdge):
        if edge in edge_list:
            raise EdgeAlreadyExistsException(edge)
        return SchemaEdgeList(frozenset(edge_list).union([edge]))

    @classmethod
    def replace_edge(cls, edge_list, edge: SchemaEdge):
        if edge not in edge_list:
            raise EdgeDoesNotExistException(edge)
        return SchemaEdgeList(frozenset(edge_list).difference([edge]).union([edge]))

    def get_edge_list(self):
        return list(self.edge_list)

    def __len__(self):
        return len(self.edge_list)

    def __getitem__(self, item):
        return list(self.edge_list)[item]

    def __repr__(self):
        return "\n".join(list(map(str, self.edge_list)))

    def __str__(self):
        return self.__repr__()
