from __future__ import annotations

import functools
import operator

import numpy as np

from frontend.derivation.exceptions import *
from frontend.tables.helpers.compose_cardinality import compose_cardinality
from representation.helpers.get_hidden_keys_in_representation import get_hidden_keys_in_representation
from representation.helpers.show_key_in_representation import show_key_in_representation
from schema.node import SchemaNode
from schema.cardinality import Cardinality
from schema.helpers.is_sublist import is_sublist
from schema.helpers.find_index import find_index
from representation.helpers.invert_representation import invert_representation
from frontend.tables.column_type import ColumnType, Key, Val, HiddenKey
from frontend.derivation.ordered_set import OrderedSet
from frontend.tables.helpers.rename_column_in_representation import (
    rename_column_in_representation,
)
from frontend.tables.helpers.transform_step import rename_hidden_keys_in_representation
from representation.representation import *
from frontend.domain import Domain


def create_value(domain: Domain, repr, hidden_keys: list[Domain], cardinality):
    return ColumnNode(domain, Val(), repr, hidden_keys, cardinality=cardinality)


def intermediate_representation_for_path(
    path: list[DerivationNode],
) -> list[RepresentationStep]:
    """
    Computes the intermediate representation for a path from keys

    Args:
        path (list[DerivationNode]): The path
    """

    intermediate_representation = []

    for node in path:
        intermediate_representation += node.intermediate_representation

    return intermediate_representation


def invert_derivation_path(
    path: list[DerivationNode], namespace: frozenset[str]
) -> DerivationNode:
    """
    Inverts a derivation path

    Args:
        path (list[DerivationNode]): The path to be inverted
        namespace (frozenset[str]): The namespace of the table with the path to be inverted removed

    Returns:
        DerivationNode: The inverted path
    """
    start: DerivationNode = path[-1]

    curr: DerivationNode = path[-2].copy()

    intermediate_representation, namespace = invert_representation(
        start.intermediate_representation,
        namespace
    )

    hidden_keys = get_hidden_keys_in_representation(intermediate_representation)

    start = start.set_parent(None)
    start = start.set_intermediate_representation([Get(start.domains)])

    new_path = start
    parent = start

    for i in range(len(path) - 2, -1, -1):
        child = None
        if i > 0:
            child = path[i - 1]
        curr = curr.set_children([c for c in curr.children if c != parent])

        inverted, namespace = invert_representation(curr.intermediate_representation, namespace)
        curr.hidden_keys = hidden_keys
        hidden_keys = get_hidden_keys_in_representation(inverted)
        curr.intermediate_representation = intermediate_representation
        intermediate_representation = inverted

        if curr.is_key_column():
            curr = curr.to_value_column(Cardinality.MANY_TO_MANY)

        new_path = new_path.add_child(parent, curr)
        parent = curr
        curr = child
        i -= 1
    return new_path

def set_and_name_hidden_keys_along_path(
    path: list[DerivationNode], entry: DerivationNode, namespace: frozenset[str]
) -> DerivationNode:
    """
    Names and sets the hidden keys along a path

    Args:
        path (list[DerivationNode]): The path
        entry (DerivationNode): The entry node
        namespace (set[str]): The namespace of the table, excluding the path

    Returns:
        DerivationNode: The entry node to the path, where the hidden keys have been named and set
    """
    path = path[1:]

    representation = entry.intermediate_representation
    new_representation, hidden_keys, new_namespace = (
        rename_hidden_keys_in_representation(namespace, representation)
    )
    node = entry.set_intermediate_representation(new_representation)
    node = node.set_hidden_keys(hidden_keys)

    if len(path) == 0:
        return node

    child = path[0]
    child = node.find_node_with_domains(child.domains)
    new_child = set_and_name_hidden_keys_along_path(path, child, new_namespace)

    node = node.remove_child(node, child).add_child(node, new_child)
    return node


def find_splice_point(
    node: DerivationNode, path: list[DerivationNode], idx: int = 0
) -> int:
    """
    Finds the splice point of a path in a tree

    Args:
        node (DerivationNode): The node to search for the splice point
        path (list[DerivationNode]): The to be spliced in at the splice point
        idx (int): Internal - the index in the list
    Returns:
        int: The point to splice in the path
    """
    if len(path) < 2:
        raise PathShouldDivergeException()

    # If I can find a child with the same domains as the first node in the path, then I should continue
    if path[1] in set(node.children):
        child = node.find_node_with_domains(path[1].domains)
        return find_splice_point(child, path[1:], idx + 1)
    else:
        return idx


class DerivationNode:
    def __init__(
        self,
        domains: list[Domain],
        intermediate_representation: list[RepresentationStep],
        hidden_keys: list[Domain] | OrderedSet = None,
        parent: DerivationNode | None = None,
        children: list[DerivationNode] | OrderedSet[DerivationNode] = None,
    ):
        self.domains: list[Domain] = domains.copy()
        self.intermediate_representation: list[RepresentationStep] = (
            intermediate_representation.copy()
        )
        self.parent: DerivationNode | None = parent
        if children is None:
            self.children: OrderedSet[DerivationNode] = OrderedSet([])
        else:
            self.children: OrderedSet[DerivationNode] = OrderedSet(children)
        if hidden_keys is None:
            self.hidden_keys: OrderedSet[DerivationNode] = OrderedSet([])
        else:
            self.hidden_keys: OrderedSet[DerivationNode] = OrderedSet(
                hidden_keys.copy()
            )

    @classmethod
    def create_root(cls, keys: list[Domain]) -> RootNode:
        """
        Creates the root of a derivation tree, given a list of unique keys

        Args:
            keys (list[Domain]): The keys of the derivation tree

        Returns:
            RootNode: The root of the derivation tree

        Raises:
            KeysMustBeUniqueException: If the keys are not unique
        """
        names = [k.name for k in keys]
        if len(names) != len(set(names)):
            raise KeysMustBeUniqueException(keys)
        root = RootNode(keys)
        return root

    def is_node_in_tree(self, node: DerivationNode) -> bool:
        """
        Checks if a node is in the tree

        Args:
            node (DerivationNode): The node to check for

        Returns:
            bool: True if the node is in the tree, False otherwise
        """
        if self == node:
            return True
        if len(self.children) == 0:
            return False
        else:
            return functools.reduce(
                operator.or_, [c.is_node_in_tree(node) for c in self.children], False
            )

    def set_parent(self, parent: DerivationNode | None) -> DerivationNode:
        """
        Sets the parent of the node

        Args:
            parent (DerivationNode): The parent to be set

        Returns:
            DerivationNode: A new node with the parent set
        """
        clone = self.copy()
        clone.parent = parent
        clone.children = OrderedSet([c.set_parent(clone) for c in self.children])
        return clone

    def set_children(
        self, children: list[DerivationNode] | OrderedSet[DerivationNode]
    ) -> DerivationNode:
        """
        Sets the children of the node

        Args:
            children (list[DerivationNode]): The children to be set

        Returns:
            DerivationNode: A new node with the children set
        """
        if isinstance(children, OrderedSet):
            children = children.to_list()

        clone = self.copy()
        clone.children = OrderedSet([c.set_parent(clone) for c in children])
        return clone

    def add_child(
        self, parent: DerivationNode, child: DerivationNode
    ) -> DerivationNode:
        """
        Adds a child to the parent node

        Args:
            parent (DerivationNode): The parent node
            child (DerivationNode): The child node

        Returns:
            DerivationNode: A new subtree where the child is added to the parent

        Raises:
            NodeIsAlreadyChildOfParentException: If the child is already a child of the parent
        """

        clone = self.copy()
        if self == parent:
            clone.children = OrderedSet([c.set_parent(clone) for c in self.children])
            if child in clone.children:
                raise NodeIsAlreadyChildOfParentException(child, parent)
            clone.children = clone.children.append(child.set_parent(clone))
            return clone
        else:
            clone = clone.set_children(
                [c.add_child(parent, child) for c in self.children]
            )
        return clone

    def add_children(
        self,
        parent: DerivationNode,
        children: list[DerivationNode] | OrderedSet[DerivationNode],
    ) -> DerivationNode:
        """
        Adds children to the parent node

        Args:
            parent (DerivationNode): The parent node
            children (list[DerivationNode]): The children nodes

        Returns:
            DerivationNode: A new subtree where the children are added to the parent

        Raises:
            NodeIsAlreadyChildOfParentException: If a child is already a child of the parent
        """

        if isinstance(children, OrderedSet):
            children = children.to_list()

        clone = self.copy()
        if self == parent:
            clone.children = OrderedSet([c.set_parent(clone) for c in self.children])
            already_children = set(clone.children).intersection(children)
            if len(already_children) > 0:
                raise NodeIsAlreadyChildOfParentException(
                    list(already_children)[0], parent
                )
            clone.children = clone.children.append_all(
                [c.set_parent(clone) for c in children]
            )
            return clone
        else:
            clone = clone.set_children(
                [c.add_children(parent, children) for c in self.children]
            )
        return clone

    def merge_subtree(self, subtree: DerivationNode) -> DerivationNode:
        """
        Merges a subtree into the tree

        Args:
            subtree (DerivationNode): The subtree to be merged

        Returns:
            DerivationNode: A new tree with the subtree merged
        """
        clone = self.copy()
        if self == subtree:
            new_children = []
            for c in self.children:
                if c in subtree.children:
                    to_merge = subtree.find_node_with_domains(c.domains)
                    new_children += [c.merge_subtree(to_merge)]
                else:
                    new_children += [c]
            for c in subtree.children:
                if c not in self.children:
                    new_children += [c]
            return clone.set_children(new_children)
        else:
            return clone.set_children([c.merge_subtree(subtree) for c in self.children])

    def remove_child(self, parent: DerivationNode, child: DerivationNode) -> DerivationNode:
        """
        Removes a child from the parent node

        Args:
            parent (DerivationNode): The parent node to remove the child from
            child (DerivationNode): The child node to be removed

        Returns:
            DerivationNode: A new subtree where the child is removed from the parent

        Raises:
            NodeIsNotChildOfParentException: If the child is not a child of the parent
        """
        parent = self.copy()
        if self == parent:
            new_children = []
            for c in self.children:
                if c == child:
                    continue
                else:
                    new_children += [c]
            parent = parent.set_children(new_children)
        else:
            parent = parent.set_children(
                [c.remove_child(parent, child) for c in self.children]
            )
        return parent

    def remove_children(
        self, parent: DerivationNode, children: list[DerivationNode] | OrderedSet[DerivationNode]
    ) -> DerivationNode:
        """
        Removes children from the parent node

        Args:
            parent (DerivationNode): The parent node to remove the child from
            children (list[DerivationNode] | OrderedSet[DerivationNode]): The children to be removed

        Returns:
            DerivationNode: A new subtree where the children are removed from the parent
        """
        if isinstance(children, OrderedSet):
            children = children.to_list()

        parent = self.copy()
        if self == parent:
            new_children = []
            for c in self.children:
                if c in set(children):
                    continue
                else:
                    new_children += [c]
            parent = parent.set_children(new_children)
        else:
            parent = parent.set_children(
                [c.remove_children(parent, children) for c in self.children]
            )
        return parent

    def set_hidden_keys(
        self, hidden_keys: list[Domain] | OrderedSet[Domain]
    ) -> DerivationNode:
        """
        Sets the hidden keys of the node

        Args:
            hidden_keys (list[Domain] | OrderedSet[Domain]): The hidden keys to be set

        Returns:
            DerivationNode: A new node with the hidden keys set
        """
        if isinstance(hidden_keys, OrderedSet):
            hidden_keys = hidden_keys.to_list()

        clone = self.copy()
        clone.hidden_keys = OrderedSet(hidden_keys)
        return clone

    def set_intermediate_representation(
        self, intermediate_representation: list[RepresentationStep]
    ):
        clone = self.copy()
        clone.intermediate_representation = intermediate_representation
        return clone

    def find_node_with_domains(self, domains: list[Domain]) -> DerivationNode | None:
        """
        Finds a node with the given domains

        Args:
            domains (list[Domain]): The domains to search for

        Returns:
            DerivationNode | None: The node with the given domains, or None if it does not exist
        """

        if self.domains == domains and self.parent is not None:
            return self
        idx = find_index(domains, [c.domains for c in self.children])
        if idx >= 0:
            child = self.children[idx]
            return child
        else:
            for child in self.children:
                poss = child.find_node_with_domains(domains)
                if poss is not None:
                    return poss

    def find_root_of_tree(self) -> RootNode:
        """
        Finds the root of the tree, given a node in the tree

        Returns:
            RootNode: The root of the tree
        """
        if self.is_root():
            assert isinstance(self, RootNode)
            return self
        else:
            return self.parent.find_root_of_tree()

    def to_intermediate_representation(self) -> list[RepresentationStep]:
        """
        Converts the subtree into an intermediate representation:
        a list of commands for populating the tree

        Returns:
            list[RepresentationStep]: The intermediate representation
        """
        ir = self.intermediate_representation.copy()  # push a frame onto the stack
        for i, child in enumerate(self.children):
            if i == 0:
                ir += [Call()] + child.to_intermediate_representation()
            else:
                ir += (
                    [Reset()] + child.to_intermediate_representation() + [Merge()]
                )  # outer merge
        if len(self.children) > 0:
            ir += [Return()]  # right merge
        return ir

    def hide(self, column: Domain) -> DerivationNode:
        """
        Hides a column in the subtree

        Args:
            column (Domain): The column to be hidden

        Returns:
            DerivationNode: A new subtree with the column hidden
        """
        idx = find_index(column, self.domains)

        if idx >= 0:
            new_node = self.copy()
            new_columns = new_node.domains[:idx] + new_node.domains[idx + 1 :]
            new_node.domains = new_columns

            if len(new_columns) == 0:
                new_node = new_node.to_derivation_node()

            for child in self.children:
                new_child = child.hide(column)
                old_columns = self.domains
                if len(new_columns) == 0:
                    intermediate_representation = [Pop(), Get(old_columns)]
                else:
                    start_node = SchemaNode.product([d.node for d in new_columns])
                    end_node = SchemaNode.product([d.node for d in old_columns])
                    indices = [i for i in range(len(old_columns)) if i != idx]
                    intermediate_representation = [
                        StartTraversal(new_columns),
                        Expand(start_node, end_node, indices, [column]),
                        EndTraversal(old_columns),
                    ]
                intermediate_node = DerivationNode(
                    old_columns, intermediate_representation, [column]
                )
                new_node = new_node.set_children([intermediate_node])
                new_node = new_node.add_child(intermediate_node, new_child)
        else:
            new_node = self.copy()
            new_children = [c.hide(column) for c in self.children]
            new_node = new_node.set_children(new_children)
        return new_node

    def show_key(self, column: Domain) -> [DerivationNode]:
        """
        Shows a key column in the subtree

        Args:
            column (Domain): The column to be shown

        Returns:
            DerivationNode: A new subtree with the column shown
        """

        without_col = []
        without_col_idxs = {}

        with_col = []
        with_col_idxs = {}

        for child in self.children:
            if column not in child.hidden_keys:
                new_children = child.show_key(column)
                for new_child in new_children:
                    idx = find_index(column, new_child.domains)
                    if idx < 0:
                        array = without_col
                        idxs = without_col_idxs
                    else:
                        array = with_col
                        idxs = with_col_idxs
                    if new_child not in idxs:
                        idxs[new_child] = len(array)
                        array += [new_child]
                    else:
                        base = array[idxs[new_child]]
                        array[idxs[new_child]] = base.add_children(
                            base, new_child.children
                        )
            else:
                new_child = child.copy()
                new_child.hidden_keys = new_child.hidden_keys.remove(column)
                new_ir = show_key_in_representation(column, new_child.intermediate_representation)
                new_child = new_child.set_intermediate_representation(new_ir)
                array = with_col
                idxs = with_col_idxs
                if new_child not in idxs:
                    idxs[new_child] = len(array)
                    array += [new_child]
                else:
                    base = array[idxs[new_child]]
                    array[idxs[new_child]] = base.add_children(base, new_child.children)

        if len(with_col) == 0:
            new_node = self.copy()
            new_node.children = OrderedSet(
                [c.set_parent(new_node) for c in without_col]
            )
            return [new_node]
        else:
            with_hk = self.copy()
            without_hk = self.copy()

            if with_hk.is_key_column():
                with_hk = with_hk.to_derivation_node()

            with_hk.domains += [column]
            with_hk.children = OrderedSet([])
            without_hk.children = OrderedSet([])
            for c in with_col:
                if c.domains != with_hk.domains:
                    with_hk = with_hk.add_child(with_hk, c)
                else:
                    with_hk = with_hk.add_children(with_hk, c.children)

            for c in without_col:
                if c.domains != without_hk.domains:
                    without_hk = without_hk.add_child(without_hk, c)
                else:
                    without_hk = without_hk.add_children(without_hk, c.children)

            return [with_hk, without_hk]

    def show_val(self, column: Domain):
        if len(self.domains) == 1 and self.domains[0] == column:
            if len(self.find_hidden_keys()) > 0:
                return self.to_value_column(Cardinality.MANY_TO_MANY)
            return self.to_value_column(Cardinality.MANY_TO_ONE)
        else:
            clone = self.copy()
            return clone.set_children([c.show_val(column) for c in self.children])

    def append_to_intermediate_representation(
        self, intermediate_representation: list[RepresentationStep]
    ) -> DerivationNode:
        """
        Appends to the intermediate representation of the subtree

        Args:
            intermediate_representation (list[RepresentationStep]): The intermediate representation to be appended
        """
        clone = self.copy()
        clone.intermediate_representation += intermediate_representation
        return clone

    def prepend_to_intermediate_representation(
        self, intermediate_representation: list[RepresentationStep]
    ) -> DerivationNode:
        """
        Prepends to the intermediate representation of the subtree

        Args:
            intermediate_representation (list[RepresentationStep]): The intermediate representation to be prepended
        """
        clone = self.copy()
        clone.intermediate_representation = (
            intermediate_representation + clone.intermediate_representation
        )
        return clone

    def equate_internal(
        self, key1: Domain, key2: Domain, root_keys: list[Domain]
    ) -> DerivationNode:
        idx1 = find_index(key1, self.domains)
        idx2 = find_index(key2, self.domains)

        new_node = self.copy()
        new_node = new_node.set_children(
            [c.equate_internal(key1, key2, root_keys) for c in self.children]
        )

        if idx2 >= 0:
            if idx1 >= 0:
                new_node.domains = self.domains[:idx2] + self.domains[idx2 + 1 :]
                start_node = SchemaNode.product([d.node for d in new_node.domains])
                end_node = key1.node
                ir = [
                    StartTraversal(new_node.domains),
                    Project(start_node, end_node, [idx1]),
                    EndTraversal([key2]),
                ]
                new_node = new_node.set_children(
                    [
                        c.prepend_to_intermediate_representation(
                            ir
                        ).append_to_intermediate_representation([Drop([key2])])
                        for c in self.children
                    ]
                )
            else:
                filtered = self.domains[:idx2] + self.domains[idx2 + 1 :]
                idxs = [(i, find_index(d, root_keys)) for i, d in enumerate(filtered)]
                idxs = [i if i[1] >= 0 else (i[0], len(root_keys)) for i in idxs]
                marker = find_index(key1, root_keys)

                def find_insertion_point(prev, curr):
                    i, j = curr
                    if j <= marker:
                        return i + 1
                    else:
                        return prev

                to_insert = functools.reduce(find_insertion_point, idxs, 0)
                new_domains = filtered[:to_insert] + [key1] + filtered[to_insert:]
                new_node.domains = new_domains
                start_node = SchemaNode.product([d.node for d in new_node.domains])
                end_node = key1.node
                ir = [
                    StartTraversal(new_node.domains),
                    Project(start_node, end_node, [to_insert]),
                    EndTraversal([key2]),
                ]
                new_node = new_node.set_children(
                    [
                        c.prepend_to_intermediate_representation(
                            ir
                        ).append_to_intermediate_representation([Drop([key2])])
                        for c in self.children
                    ]
                )

        return new_node

    def rename(self, old_name: str, new_name: str) -> DerivationNode:
        """
        Renames a domain in the subtree, as well as in the intermediate representation

        Args:
            old_name (str): The old name of the domain
            new_name (str): The new name of the domain

        Returns:
            DerivationNode: A new subtree with the domain renamed
        """

        idx = find_index(old_name, [d.name for d in self.domains])
        if idx >= 0:
            old_domain = self.domains[idx]
            new_domain = Domain(new_name, old_domain.node)
            new_domains = self.domains[:idx] + [new_domain] + self.domains[idx + 1 :]
        else:
            new_domains = self.domains
        new_node = self.copy()
        new_node.domains = new_domains.copy()
        new_node.intermediate_representation = rename_column_in_representation(
            self.intermediate_representation, old_name, new_name
        )
        new_node = new_node.set_children(
            [c.rename(old_name, new_name) for c in self.children]
        )
        return new_node

    def path_to_value(self, value: DerivationNode) -> list[DerivationNode] | None:
        """
        Finds the path from the node to a value column

        Args:
            value (DerivationNode): The value column to find the path to

        Returns:
            list[DerivationNode] | None: The path from the node to the value column, or None if it does not exist
        """
        if self == value:
            if self.is_value_or_set_of_values():
                return [self]
            else:
                return None
        else:
            for child in self.children:
                suffix = child.path_to_value(value)
                if suffix is not None:
                    return [self] + suffix
            return None

    def find_column_with_name(self, name: str) -> ColumnNode | None:
        """
        Finds a column with a given name in the subtree

        Args:
            name (str): The name of the column to find

        Returns:
            ColumnNode | None: The column with the given name, or None if it does not exist
        """
        if isinstance(self, ColumnNode):
            if self.domains[0].name == name:
                return self
        for child in self.children:
            found = child.find_column_with_name(name)
            if found is not None:
                return found

    def find_all_keys_in_tree(self) -> list[ColumnNode]:
        """
        Finds all key columns in the subtree

        Returns:
            list[Domain]: The key columns in the subtree
        """
        res = []
        if self.is_key_column():
            res += [self]
        for child in self.children:
            res += child.find_all_keys_in_tree()
        return res

    def find_all_values_in_tree(self) -> list[ColumnNode]:
        """
        Finds all value columns in the subtree

        Returns:
            list[Domain]: The value columns in the subtree
        """
        res = []
        if self.is_val_column():
            res += [self]
        for child in self.children:
            res += child.find_all_values_in_tree()
        return res

    def find_all_hidden_keys_in_tree(self) -> list[ColumnNode]:
        """
        Finds all hidden key columns in the subtree

        Returns:
            list[Domain]: The hidden key columns in the subtree
        """
        res = []
        if self.is_hidden_key_column():
            res += [self]
        for child in self.children:
            res += child.find_all_hidden_keys_in_tree()
        return res

    def find_strong_keys(self) -> list[Domain]:
        if self.parent.is_root():
            return self.domains
        else:
            return self.parent.find_strong_keys()

    def find_hidden_keys(self) -> OrderedSet:
        return self.parent.find_hidden_keys().union(self.hidden_keys)

    def is_root(self) -> bool:
        return False

    def is_key_column(self) -> bool:
        return False

    def is_hidden_key_column(self) -> bool:
        return False

    def is_val_column(self) -> bool:
        return False

    def is_intermediate_node(self) -> bool:
        return False

    def is_value_or_set_of_values(self) -> bool:
        """
        Checks if the node is a value column or a set of value columns
        A node represents a set of value columns when it is of the form A x B x C x ... where A, B, C, ... are value columns

        Returns:
            bool: True if the node is a value column or a set of value columns, False otherwise
        """
        if self.is_val_column():
            return True
        count = 0
        for domain in self.domains:
            idx = find_index([domain], [c.domains for c in self.children.item_list])
            if idx < 0:
                return False
            child = self.children[idx]
            if not child.is_val_column():
                return False
            count += 1
        return count == len(self.children)

    def is_key_or_set_of_keys(self) -> bool:
        """
        Checks if the node is a key column or a set of key columns
        A node represents a set of key columns when it is of the form A x B x C x ... where A, B, C, ... are key columns

        Returns:
            bool: True if the node is a key column or a set of key columns, False otherwise
        """

        return self.is_key_column() or self.parent.is_root()

    def to_derivation_node(self) -> DerivationNode:
        copy = DerivationNode(
            self.domains, self.intermediate_representation, self.hidden_keys
        )
        copy.parent = self.parent
        copy = copy.add_children(copy, self.children)
        return copy

    def to_key_column(self) -> ColumnNode:
        copy = ColumnNode(
            self.domains[0], Key(), self.intermediate_representation, self.hidden_keys
        )
        copy.parent = self.parent
        copy = copy.add_children(copy, self.children)
        return copy

    def to_hidden_key_column(self) -> ColumnNode:
        copy = ColumnNode(
            self.domains[0],
            HiddenKey(),
            self.intermediate_representation,
            self.hidden_keys,
        )
        copy.parent = self.parent
        copy = copy.add_children(copy, self.children)
        return copy

    def to_value_column(self, cardinality) -> ColumnNode:
        copy = ColumnNode(
            self.domains[0],
            Val(),
            self.intermediate_representation,
            self.hidden_keys,
            cardinality,
        )
        copy.parent = self.parent
        copy = copy.add_children(copy, self.children)
        return copy

    def to_intermediate_node(self) -> IntermediateNode:
        intermediate = IntermediateNode(
            self.domains, self.intermediate_representation, self.hidden_keys
        )
        intermediate.parent = self.parent
        intermediate = intermediate.add_children(intermediate, self.children)
        return intermediate

    def __hash__(self):
        return hash((tuple([c.name for c in self.domains])))

    def __eq__(self, other):
        if isinstance(other, DerivationNode):
            return self.domains == other.domains
        raise NotImplemented()

    def __repr__(self):
        child_repr = ""
        for child in self.children:
            child_repr += "\n" + "\t" + child.__repr__().replace("\n", "\n\t")
        return f"{self.domains} // hidden: {self.hidden_keys.item_list}" + child_repr

    def __str__(self):
        return self.__repr__()

    def copy(self) -> DerivationNode:
        copy = DerivationNode(
            self.domains, self.intermediate_representation, self.hidden_keys
        )
        copy.parent = self.parent
        copy.children = self.children
        return copy


def compress_path_representation(
    path: list[DerivationNode],
) -> list[RepresentationStep]:
    """
    Compresses the intermediate representation of a path

    Args:
        path (list[DerivationNode]): The path to be compressed

    Returns:
        list[RepresentationStep]: The compressed intermediate representation
    """
    val_node = path[-1]
    keys = val_node.find_strong_keys()
    hidden_keys = val_node.find_hidden_keys()

    intermediate_repr = intermediate_representation_for_path(path)

    ents = [step for step in intermediate_repr if isinstance(step, EndTraversal)]
    to_drop = functools.reduce(
        lambda x, y: x.union(y), [step.end_columns for step in ents[:-1]], set()
    )
    to_keep = set(ents[-1].end_columns).union(set(hidden_keys)).union(set(keys))

    return intermediate_repr + [Drop(list(to_drop.difference(to_keep)))]


class RootNode(DerivationNode):

    def __init__(self, keys: list[Domain]):
        super().__init__(keys, [])

        unit_node = DerivationNode([], [Get([])], [])
        self.children = self.children.append(unit_node)

        for i, key in enumerate(keys):
            key_node = ColumnNode(key, Key(), [Get([key])], parent=self)
            self.children = self.children.append(key_node)

    def merge_subtree(self, subtree: DerivationNode) -> DerivationNode:
        new_children = []
        for child in self.children:
            if child == subtree:
                new_children += [child.merge_subtree(subtree)]
            else:
                new_children += [child]
        return self.set_children(new_children)

    def get_representation_of_values_from_keys(
        self, keys: list[Domain], values: list[Domain]
    ) -> list[RepresentationStep]:
        """
        Returns the intermediate representation -- a list of imperative commands -- for a path from keys to values

        Args:
            keys (list[Domain]): The keys
            values (list[Domain]): The values

        Returns:
            list[RepresentationStep]: The intermediate representation
        """
        key_node = self.find_node_with_domains(keys)
        if key_node is None:
            raise KeysNotFoundException(keys)

        val_node = self.find_node_with_domains(values)
        if val_node is None:
            raise ValuesNotFoundException(values)

        path = key_node.path_to_value(val_node)
        if path is None:
            raise PathNotFoundException(keys, values)

        path = path[1:]

        return compress_path_representation(path)

    def hide(self, column: ColumnNode) -> RootNode:
        # assert column.is_key_column()
        idx = find_index(column.get_domain(), self.domains)
        new_root = RootNode(self.domains[:idx] + self.domains[idx + 1 :])
        for child in self.children:
            new_child = child.hide(column.get_domain())
            new_root = new_root.insert_key(new_child.domains)
            new_root = new_root.merge_subtree(new_child)

        if column.is_key_column():
            new_root = new_root.add_hidden_key(column.get_domain())

        return new_root

    def show(self, column: Domain) -> RootNode:
        node = self.find_node_with_domains([column])
        if node is None:
            return self
        elif node.is_hidden_key_column():
            return self.show_key(column)
        else:
            return self.show_val(column)

    def show_key(self, column: Domain) -> RootNode:
        new_root = RootNode(self.domains + [column])
        for child in self.children:
            new_children = child.show_key(column)
            for new_child in new_children:
                new_root = new_root.insert_key(new_child.domains)
                key_node = new_root.find_node_with_domains(new_child.domains)
                new_root = new_root.add_children(key_node, new_child.children)
        new_root = new_root.remove_hidden_key(column)
        return new_root

    def show_val(self, column: Domain) -> RootNode:
        return self.set_children([c.show_val(column) for c in self.children])

    # inner product
    def equate(self, key1, key2):
        key1 = key1.get_domain()
        key2 = key2.get_domain()
        idx = find_index(key2, self.domains)
        new_keys = self.domains[:idx] + self.domains[idx + 1 :]
        new_root = RootNode(new_keys)
        for child in self.children:
            new_child = child.equate_internal(key1, key2, new_keys)
            new_root = new_root.insert_key(new_child.domains)
            key_node = new_root.find_node_with_domains(new_child.domains)
            new_root = new_root.add_children(key_node, new_child.children)
        return new_root

    def splice(self, path: list[DerivationNode], namespace: frozenset[str]):
        root = self.insert_key(path[0].domains)
        key_node = root.find_node_with_domains(path[0].domains)
        splice_point_idx = find_splice_point(key_node, path)
        splice_point = root.find_node_with_domains(path[splice_point_idx].domains)

        intermediate_representation = compress_path_representation(path[splice_point_idx+1:])
        target = path[-1]
        target = target.set_intermediate_representation(intermediate_representation)

        to_splice_in = set_and_name_hidden_keys_along_path(
            [target], target, namespace
        )

        hidden_keys = to_splice_in.find_hidden_keys()
        to_splice_in.hidden_keys = hidden_keys
        root = root.add_hidden_keys(hidden_keys)
        root = root.add_child(splice_point, to_splice_in)
        return root

    def infer(
        self,
        value: Domain,
        strong_keys: list[Domain],
        old_hids: list[Domain],
        new_hids: list[Domain],
        intermediates: list,
        cardinality,
        repr: list[RepresentationStep],
    ):
        to_prepend = []
        filtered = set([i.domains[0] for i in intermediates]).difference(
            set(strong_keys)
        )
        new_root = self.insert_key(strong_keys)
        new_root = new_root.add_hidden_keys(new_hids)
        key_node = new_root.find_node_with_domains(strong_keys)
        val_node = create_value(value, repr, new_hids, cardinality)
        j = 0
        if len(filtered) > 0:
            for i, intermediate in enumerate(intermediates):
                if intermediate.is_val_column():
                    i_keys = intermediate.get_strong_keys()
                    key_node = self.find_node_with_domains(i_keys)
                    intermediate_steps = self.get_representation_of_values_from_keys(
                        i_keys, intermediate.domains
                    )
                    if j == 0:
                        to_prepend += intermediate_steps
                    else:
                        to_prepend += [Reset()] + intermediate_steps + [Merge()]
                    j += 1
            intermediate_node = new_root.find_node_with_domains(
                [i.domains[0] for i in intermediates]
            )
            if intermediate_node is None:
                intermediate_cardinality = functools.reduce(
                    compose_cardinality, [i.cardinality for i in intermediates]
                )
                intermediate_node = IntermediateNode(
                    [i.domains[0] for i in intermediates],
                    to_prepend,
                    old_hids,
                    cardinality=intermediate_cardinality,
                )
                new_root = new_root.add_child(key_node, intermediate_node)
            new_root = new_root.add_child(intermediate_node, val_node)
        else:
            # TODO: if no key and single hidden key
            new_root = new_root.add_child(key_node, val_node)
        return new_root

    def compose(
        self, new_keys: list[Domain], old_key: Domain, hidden_keys, repr, cardinality
    ):
        old_idx = find_index(old_key, self.domains)
        new_root = DerivationNode.create_root(
            self.domains[:old_idx] + new_keys + self.domains[old_idx + 1 :]
        )
        for child in self.children:
            if child.domains == [old_key]:
                if len(child.children) == 0:
                    continue
                elif old_key in set(new_keys):
                    new_keys_node = new_root.find_node_with_domains([old_key])
                    new_root = new_root.add_children(new_keys_node, child.children)
                else:
                    new_root = new_root.insert_key(new_keys)
                    new_keys_node = new_root.find_node_with_domains(new_keys)
                    intermediate = IntermediateNode(
                        [old_key], repr, hidden_keys, None, [], cardinality
                    )
                    new_root = new_root.add_child(
                        new_keys_node, intermediate
                    ).add_children(intermediate, child.children)
            else:
                domains = child.domains
                idx = find_index(old_key, domains)

                if idx >= 0:
                    new_domains = domains[:idx] + new_keys + domains[idx + 1 :]
                    new_root = new_root.insert_key(new_domains)
                    new_key_node = new_root.find_node_with_domains(new_domains)
                    intermediate = IntermediateNode(
                        domains, repr, child.hidden_keys + hidden_keys
                    )
                    new_root = new_root.add_child(
                        new_key_node, intermediate
                    ).add_children(intermediate, child.children)

                else:
                    new_root = new_root.insert_key(child.domains)
                    key_node = new_root.find_node_with_domains(child.domains)
                    new_root = new_root.add_children(key_node, child.children)

        return new_root

    def rename(self, old_name: str, new_name: str):
        idx = find_index(old_name, [d.name for d in self.domains])
        if idx >= 0:
            old_domain = self.domains[idx]
            new_domain = Domain(new_name, old_domain.node)
            new_domains = self.domains[:idx] + [new_domain] + self.domains[idx + 1 :]
            new_root = RootNode(new_domains)
        else:
            new_root = RootNode(self.domains)

        for child in self.children:
            new_child = child.rename(old_name, new_name)
            new_root = new_root.insert_key(new_child.domains)
            key_node = new_root.find_node_with_domains(new_child.domains)
            new_root = new_root.add_children(key_node, new_child.children)
        return new_root

    def is_node_in_tree(self, node):
        return functools.reduce(
            operator.or_, [c.is_node_in_tree(node) for c in self.children], False
        )

    def get_values(self) -> list[ColumnNode]:
        return self.find_all_values_in_tree()

    def get_keys(self) -> list[ColumnNode]:
        return self.find_all_keys_in_tree()

    def get_keys_and_values(self) -> list[ColumnNode]:
        keys = self.get_keys()
        values = self.get_values()
        return keys + values

    def get_hidden(self) -> list[ColumnNode]:
        return self.find_all_hidden_keys_in_tree()

    def find_hidden_keys(self):
        return OrderedSet([])

    def insert_key(self, domains: list[Domain]):
        assert is_sublist(domains, self.domains)
        is_exact_match = np.sum(
            np.array([child.domains == domains for child in self.children])
        )

        if is_exact_match == 1:
            return self
        else:
            assert is_exact_match == 0
            node = DerivationNode(domains, [Get(domains)], [])
            new_root = self.copy()
            new_root.children = OrderedSet(
                [c.set_parent(new_root) for c in self.children]
            )
            new_root.children = new_root.children.append(node.set_parent(new_root))
            return new_root

    def add_hidden_key(self, hidden_key: Domain):
        new_root = self.copy()
        new_root.children = OrderedSet([c.set_parent(new_root) for c in self.children])
        unit_key = new_root.find_node_with_domains([])
        hidden_key_node = ColumnNode(
            hidden_key,
            HiddenKey(),
            [Pop(), Get([hidden_key])],
            parent=new_root,
            hidden_keys=[hidden_key],
        )
        unit_key = unit_key.set_children([hidden_key_node])
        new_root = new_root.merge_subtree(unit_key)
        return new_root

    def remove_hidden_key(self, hidden_key: Domain) -> RootNode:
        new_root = self.copy()
        new_root.children = OrderedSet([c.set_parent(new_root) for c in self.children])
        unit_key = new_root.find_node_with_domains([])
        hidden_key_node = new_root.find_node_with_domains([hidden_key])
        new_root = new_root.remove_child(unit_key, hidden_key_node)
        return new_root

    def add_hidden_keys(self, hidden_keys: list[Domain] | OrderedSet[Domain]):
        if isinstance(hidden_keys, OrderedSet):
            hidden_keys = hidden_keys.to_list()

        new_root = self.copy()
        new_root.children = OrderedSet([c.set_parent(new_root) for c in self.children])
        unit_key = new_root.find_node_with_domains([])
        new_children = []
        for hidden_key in hidden_keys:
            hidden_key_node = ColumnNode(
                hidden_key, HiddenKey(), [Pop(), Get([hidden_key])], parent=new_root
            )
            new_children += [hidden_key_node]
        unit_key = unit_key.set_children(new_children)
        new_root = new_root.merge_subtree(unit_key)
        return new_root

    def add_child(self, parent, child) -> RootNode:
        clone = self.copy()
        clone.children = OrderedSet(
            [c.set_parent(clone).add_child(parent, child) for c in self.children]
        )
        return clone

    def add_children(self, parent, children):
        clone = self.copy()
        clone.children = OrderedSet(
            [c.set_parent(clone).add_children(parent, children) for c in self.children]
        )
        return clone

    def remove_child(self, parent, child) -> RootNode:
        clone = self.copy()
        clone.children = OrderedSet(
            [c.set_parent(clone).remove_child(parent, child) for c in self.children]
        )
        return clone

    def remove_children(self, parent, children):
        clone = self.copy()
        clone.children = OrderedSet(
            [c.set_parent(clone).remove_children(parent, children) for c in self.children]
        )
        return clone

    def is_root(self):
        return True

    def to_intermediate_representation(self):
        ir = self.intermediate_representation.copy()  # push a frame onto the stack
        for i, child in enumerate(self.children):
            if i == 0:
                ir += child.to_intermediate_representation()
            else:
                ir += child.to_intermediate_representation() + [Merge()]  # outer merge
        return ir

    def copy(self) -> RootNode:
        return RootNode(self.domains)


class ColumnNode(DerivationNode):
    def __init__(
        self,
        column: Domain,
        column_type: ColumnType,
        intermediate_representation: list[RepresentationStep],
        hidden_keys: list[Domain] | OrderedSet = None,
        parent=None,
        cardinality=None,
    ):
        super().__init__(
            [column],
            intermediate_representation,
            hidden_keys=hidden_keys,
            parent=parent,
        )
        self.name = column.name
        self.column_type = column_type
        self.cardinality = cardinality

    def is_key_column(self):
        return self.column_type.is_key_column()

    def is_val_column(self):
        return self.column_type.is_val_column()

    def is_hidden_key_column(self):
        return self.column_type.is_hidden_key_column()

    def get_strong_keys(self):
        return self.column_type.get_strong_keys(self)

    def get_hidden_keys(self):
        return self.column_type.get_hidden_keys(self)

    def get_derivation(self):
        return self.column_type.get_derivation(self)

    def get_schema_node(self):
        return self.domains[0].node

    def get_name(self):
        return self.domains[0].name

    def set_cardinality(self, cardinality):
        self.cardinality = cardinality

    def get_cardinality(self):
        return self.cardinality

    def get_domain(self):
        return self.domains[0]

    def copy(self):
        copy = ColumnNode(
            self.domains[0],
            self.column_type,
            self.intermediate_representation.copy(),
            self.hidden_keys,
            cardinality=self.cardinality,
        )
        copy.parent = self.parent
        copy.children = self.children
        return copy

    def hide(self, column: Domain) -> DerivationNode:
        new_node = super().hide(column)
        if self.get_domain() == column:
            if self.is_key_column():
                new_children = []
                for child in new_node.children:
                    if child.domains == [column]:
                        new_children += [child.to_hidden_key_column()]
                    else:
                        new_children += [child]
                return new_node.set_children(new_children)
            elif self.is_val_column():
                return self.to_derivation_node()
        else:
            return new_node

    def show_key(self, column: Domain) -> [DerivationNode]:
        if self.is_hidden_key_column():
            return [self.to_key_column()]
        else:
            return super().show_key(column)


class IntermediateNode(DerivationNode):
    def __init__(
        self,
        domains: list[Domain],
        intermediate_representation: list[RepresentationStep],
        hidden_keys: list[Domain] | OrderedSet = None,
        parent=None,
        children: list | OrderedSet = None,
        cardinality: Cardinality = None,
    ):
        super().__init__(
            domains, intermediate_representation, hidden_keys, parent, children
        )
        self.cardinality = cardinality

    def to_intermediate_representation(self):
        assert len(self.children) > 0
        assert self.parent is not None
        intermediates = self.get_intermediates()
        ir = super().to_intermediate_representation()
        return ir[:-1] + [Drop(intermediates)] + [Return()]

    def get_intermediates(self):
        root = self.find_root_of_tree()
        keys = set([n.get_domain() for n in root.get_keys()])
        vals = set([n.get_domain() for n in self.find_all_values_in_tree()])
        hids = set([n for n in self.find_hidden_keys()])
        intermediates = [c for c in self.domains if c not in keys | vals | hids]
        return intermediates

    def is_intermediate_node(self):
        return True

    def copy(self):
        copy = IntermediateNode(
            self.domains,
            self.intermediate_representation.copy(),
            self.hidden_keys,
            cardinality=self.cardinality,
        )
        copy.parent = self.parent
        copy.children = self.children
        return copy
