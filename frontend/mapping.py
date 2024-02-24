from __future__ import annotations

from typing import TypeVar, Callable

from frontend.domain import Domain
from frontend.exceptions import *
from frontend.transform import *
from schema.cardinality import Cardinality
from schema.edge import SchemaEdge, reverse_cardinality
from schema.node import SchemaNode, AtomicNode, SchemaClass

T = TypeVar('T')


def instantiate_list(raw: list[T] | None) -> list[T]:
    if raw is None:
        return []
    else:
        return raw


def weaken_cardinality(cardinality: Cardinality) -> Cardinality:
    if cardinality == Cardinality.ONE_TO_ONE:
        return Cardinality.ONE_TO_MANY
    elif cardinality == Cardinality.MANY_TO_ONE:
        return Cardinality.MANY_TO_MANY
    elif cardinality == Cardinality.ONE_TO_MANY:
        return Cardinality.MANY_TO_MANY
    else:
        return cardinality


def strengthen_cardinality(cardinality: Cardinality) -> Cardinality:
    if cardinality == Cardinality.ONE_TO_MANY:
        return Cardinality.MANY_TO_ONE
    elif cardinality == Cardinality.MANY_TO_MANY:
        return Cardinality.MANY_TO_ONE
    else:
        return cardinality


class Mapping:
    def __init__(self,
                 edge: SchemaEdge,
                 from_nodes: list[AtomicNode | SchemaClass] = None,
                 to_nodes: list[AtomicNode | SchemaClass] = None,
                 hidden_keys: list[Domain] = None,
                 cardinality: Cardinality = None,
                 transform: list[Transform] = None,
                 carried: dict[Domain, (int, int)] = None):

        self.edge: SchemaEdge = edge
        if from_nodes is None:
            self.from_nodes: list[AtomicNode | SchemaClass] = SchemaNode.get_constituents(edge.from_node)
        else:
            self.from_nodes: list[AtomicNode | SchemaClass] = from_nodes

        if to_nodes is None:
            self.to_nodes: list[AtomicNode | SchemaClass] = SchemaNode.get_constituents(edge.to_node)
        else:
            self.to_nodes = to_nodes

        if cardinality is None:
            self.cardinality: Cardinality = edge.cardinality
        else:
            self.cardinality: Cardinality = cardinality

        self.hidden_keys: list[Domain] = instantiate_list(hidden_keys)

        self.transform: list[Transform] = instantiate_list(transform)

        if carried is None:
            self.carried: dict[Domain, int] = {}
        else:
            self.carried = carried

    def get_cardinality(self) -> Cardinality:
        return self.edge.cardinality

    def get_underlying_edge(self) -> SchemaEdge:
        return self.edge

    def get_hidden_keys(self) -> list[Domain]:
        return self.hidden_keys

    def num_from_domains(self):
        return len(self.from_nodes)

    def num_to_domains(self):
        return len(self.to_nodes)

    def in_from(self, indices: list[int]) -> bool:
        return all([i in range(self.num_from_domains()) for i in indices])

    def in_to(self, indices: list[int]) -> bool:
        return all([i in range(self.num_to_domains()) for i in indices])

    def curry(self, index: int, hidden_key: Domain) -> Mapping:
        if not self.in_from([index]):
            raise MustCurryAtSourceException()
        transform = Curry(index, hidden_key)
        new_transform = self.transform + [transform]
        assert self.from_nodes[index] == hidden_key.node
        if index in set([x[0] for x in self.carried.values()]):
            domain = {i: d for d, (i, j) in self.carried.items()}[index]
            assert domain.name == hidden_key.name

        new_from = [n for i, n in enumerate(self.from_nodes) if i != index]
        new_hk = [hidden_key] + self.hidden_keys
        new_cardinality = weaken_cardinality(self.cardinality)

        if index in set([x[0] for x in self.carried.values()]):
            new_carried = {d: ((-1, j) if i == index else (i, j) if 0 <= i < index else (i-1, j)) for d, (i, j) in self.carried.items()}
        else:
            new_carried = {d: ((i, j) if 0 <= i < index else (i-1, j)) for d, (i, j) in self.carried.items()}
        return Mapping(self.edge, new_from, self.to_nodes, new_hk, new_cardinality, new_transform, new_carried)

    def uncurry(self, domain: Domain) -> Mapping:
        if domain not in set(self.hidden_keys):
            raise MustUncurryHiddenKeyException()
        idx = [i for i, d in enumerate(self.hidden_keys) if d == domain]
        assert len(idx) == 1
        idx = idx[0]
        transform = Uncurry(idx, len(self.from_nodes))
        new_transform = self.transform + [transform]
        new_from = self.from_nodes + [domain.node]
        new_hk = [d for d in self.hidden_keys if d != domain]
        if new_from == SchemaNode.get_constituents(self.edge.from_node):
            new_cardinality = self.edge.cardinality
        elif len(new_hk) == 0:
            new_cardinality = strengthen_cardinality(self.cardinality)
        else:
            new_cardinality = self.cardinality
        if domain in self.carried.keys():
            new_carried = {d: ((i, j) if d == domain else (len(self.from_nodes), j)) for d, (i, j) in self.carried.items()}
        else:
            hidx = -idx-1
            new_carried = {d: ((i, j) if i > hidx else (i+1, j)) for d, (i, j) in self.carried.items()}
        return Mapping(self.edge, new_from, self.to_nodes, new_hk, new_cardinality, new_transform, new_carried)

    def carry(self, domain: Domain) -> Mapping:
        assert domain not in self.carried
        transform = Carry(domain, len(self.from_nodes), len(self.to_nodes))
        new_transform = self.transform + [transform]
        new_from = self.from_nodes + [domain.node]
        new_to = self.to_nodes + [domain.node]

        carried = self.carried.copy()
        carried |= {domain: (len(self.from_nodes), len(self.to_nodes))}

        return Mapping(self.edge, new_from, new_to, self.hidden_keys, self.cardinality, new_transform, carried)

    def drop(self, domain: Domain) -> Mapping:
        assert domain in self.carried
        drop_from, drop_to = self.carried[domain]
        transform = Drop(drop_from, drop_to)
        new_transform = self.transform + [transform]
        new_from = [n for i, n in enumerate(self.from_nodes) if i != drop_from]
        new_to = [n for i, n in enumerate(self.to_nodes) if i != drop_to]

        carried = {d: (i-1, j-1) for d, (i, j) in self.carried.items() if d != domain}

        return Mapping(self.edge, new_from, new_to, self.hidden_keys, self.cardinality, new_transform, carried)

    def invert(self) -> Callable[[set[str], Callable[[set[str], str], str]], tuple[Mapping, set[str]]]:
        if reverse_cardinality(self.cardinality) == Cardinality.MANY_TO_ONE or reverse_cardinality(self.cardinality) == Cardinality.ONE_TO_ONE:
            new_hk = []
            to_exclude = []
        else:
            to_exclude = (set([x[0] for x in self.carried.values() if x[0] >= 0]))
            new_hk = [n for i, n in enumerate(self.from_nodes) if i not in to_exclude]

        def name_hks(namespace: set[str], naming_function: Callable[[set[str], str], str]) -> tuple[Mapping, set[str]]:
            internal_namespace = namespace.copy()
            new_hks = [Domain(naming_function(internal_namespace, d.name), d) for d in new_hk]
            internal_namespace |= set([d.name for d in new_hks])
            new_transform = self.transform + [Invert(new_hks, self.num_from_domains(), self.num_to_domains(), list(sorted(to_exclude)))]
            new_cardinality = reverse_cardinality(self.cardinality)
            new_edge = SchemaEdge.invert(self.edge)
            new_carried = {d: (j, i) for d, (i, j) in self.carried.items() if i >=0}
            mapping = Mapping(new_edge, self.to_nodes, self.from_nodes, new_hks, new_cardinality, new_transform, new_carried)
            return mapping, internal_namespace

        return name_hks

    def __repr__(self):
        return SchemaEdge(SchemaNode.product(self.from_nodes), SchemaNode.product(self.to_nodes), self.cardinality).__repr__()

    def __str__(self):
        return self.__repr__()