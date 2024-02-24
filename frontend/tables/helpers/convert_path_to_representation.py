from typing import Callable

from frontend.domain import Domain
from frontend.mapping import Mapping
from representation.representation import (
    Traverse,
    Project,
    Expand,
    RepresentationStep,
    StartTraversal,
    EndTraversal,
    Equate,
)
from schema.edge import SchemaEdge
from schema.helpers.get_indices_of_sublist import get_indices_of_sublist
from schema.helpers.is_sublist import is_sublist
from schema.node import SchemaNode

Namespace = frozenset[str]


def convert_edge_to_representation_step(
    edge: SchemaEdge, namespace: Namespace
) -> tuple[Traverse | Project | Expand | Equate, list[Domain], Namespace]:
    from frontend.tables.table import new_domain_from_schema_node

    start = edge.from_node
    end = edge.to_node
    start_nodes = SchemaNode.get_constituents(edge.from_node)
    end_nodes = SchemaNode.get_constituents(edge.to_node)

    internal_namespace = namespace

    if edge.is_equality():
        return Equate(start, end), [], namespace

    if not edge.is_functional():
        from schema.helpers.list_difference import list_difference

        if is_sublist(start_nodes, end_nodes):
            hidden_nodes = list_difference(end_nodes, start_nodes)
            hidden_keys = []
            for node in hidden_nodes:
                d = new_domain_from_schema_node(internal_namespace, node)
                hidden_keys += [d]
                internal_namespace |= {d.name}
            indices = get_indices_of_sublist(start_nodes, end_nodes)

            return (
                Expand(start, end, indices, hidden_keys),
                hidden_keys,
                frozenset(internal_namespace),
            )
        else:
            hidden_nodes = end_nodes
            hidden_keys = []
            for node in hidden_nodes:
                d = new_domain_from_schema_node(internal_namespace, node)
                hidden_keys += [d]
                internal_namespace |= {d.name}
            return (
                Traverse(Mapping.create_mapping_from_edge(edge, hidden_keys)),
                hidden_keys,
                frozenset(internal_namespace),
            )

    else:
        if is_sublist(end_nodes, start_nodes):
            indices = get_indices_of_sublist(end_nodes, start_nodes)
            return Project(start, end, indices), [], namespace
        return Traverse(Mapping.create_mapping_from_edge(edge)), [], namespace


def convert_path_to_representation(
    start_domains: list[Domain],
    end_domains: list[Domain],
    path: list[SchemaEdge],
    namespace: frozenset[str],
) -> tuple[list[RepresentationStep], list[Domain]]:
    steps = [StartTraversal(start_domains)]
    hidden_keys = []
    for edge in path:
        step, hks, namespace = convert_edge_to_representation_step(edge, namespace)
        steps += [step]
        hidden_keys += hks
    return steps + [EndTraversal(end_domains)], hidden_keys
