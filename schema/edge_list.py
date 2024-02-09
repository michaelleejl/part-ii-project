from __future__ import annotations

from collections.abc import Iterable

from schema.edge import SchemaEdge
from schema.exceptions import EdgeAlreadyExistsException, EdgeDoesNotExistException


class SchemaEdgeList(Iterable):
    """
    SchemaEdgeList represents a list of edges in a schema graph
    """

    def __init__(self, edge_list: frozenset[SchemaEdge] = frozenset()):
        self.edge_list = edge_list

    @classmethod
    def add_edge(cls, edge_list: SchemaEdgeList, edge: SchemaEdge) -> SchemaEdgeList:
        """
        Adds an edge to an edge list

        Args:
            edge_list (SchemaEdgeList): The edge list
            edge (SchemaEdge): The edge to be added

        Returns:
            SchemaEdgeList: The new edge list

        Raises:
            EdgeAlreadyExistsException: If the edge already exists in the edge list
        """

        if edge in edge_list:
            raise EdgeAlreadyExistsException(edge)
        return SchemaEdgeList(frozenset(edge_list).union([edge]))

    @classmethod
    def replace_edge(cls, edge_list: SchemaEdgeList, edge: SchemaEdge) -> SchemaEdgeList:
        """
        Replaces an edge in an edge list

        Args:
            edge_list (SchemaEdgeList): The edge list
            edge (SchemaEdge): The edge to be replaced

        Returns:
            SchemaEdgeList: The new edge list

        Raises:
            EdgeDoesNotExistException: If the edge does not exist in the edge list
        """
        if edge not in edge_list:
            raise EdgeDoesNotExistException(edge)
        return SchemaEdgeList(frozenset(edge_list).difference([edge]).union([edge]))

    def get_edge_list(self) -> list[SchemaEdge]:
        """
        Returns the internal edge list

        Returns:
            list[SchemaEdge]: The internal edge list
        """
        return list(self.edge_list)

    def __iter__(self):
        return self.edge_list.__iter__()

    def __contains__(self, item: SchemaEdge):
        return item in set(self.edge_list)

    def __len__(self):
        return len(self.edge_list)

    def __getitem__(self, item):
        return list(self.edge_list)[item]

    def __repr__(self):
        return "\n".join(list(map(str, self.edge_list)))

    def __str__(self):
        return self.__repr__()
