import functools
import operator

import numpy as np

from helpers.compose_cardinality import compose_cardinality
from schema import SchemaNode, is_sublist, Cardinality
from schema.helpers.find_index import find_index
from schema.helpers.invert_representation import invert_representation
from tables.derivation.column_type import ColumnType, Key, Val
from tables.derivation.ordered_set import OrderedSet
from tables.helpers.transform_step import transform_step
from tables.internal_representation import *
from tables.domain import Domain


class DerivationNode:
    def __init__(self, domains: list[Domain], intermediate_representation: list[RepresentationStep],
                 hidden_keys: list[Domain] | OrderedSet = None, parent=None,
                 children: list | OrderedSet = None):
        self.domains = domains.copy()
        self.intermediate_representation = intermediate_representation.copy()
        self.on_completion = None
        self.parent = parent
        if children is None:
            self.children = OrderedSet([])
        else:
            self.children = OrderedSet(children)
        if hidden_keys is None:
            self.hidden_keys = OrderedSet([])
        else:
            self.hidden_keys = OrderedSet(hidden_keys.copy())

    @classmethod
    def create_root(cls, domains: list[Domain]):
        root = RootNode(domains)
        return root

    def get_all_hidden_keys_in_subtree(self):
        hidden_keys = OrderedSet([])
        for child in self.children:
            hk = child.get_all_hidden_keys_in_subtree()
            hidden_keys = hidden_keys.union(hk)
        hidden_keys = hidden_keys.union(self.hidden_keys)
        return hidden_keys

    def is_node_in_tree(self, node):
        if self == node:
            return True
        if len(self.children) == 0:
            return False
        else:
            return functools.reduce(operator.or_, [c.is_node_in_tree(node) for c in self.children])

    def set_parent(self, parent):
        self.parent = parent
        return self

    def add_child(self, parent, child):
        clone = self.copy()
        if self == parent:
            clone.children = OrderedSet([c.set_parent(clone) for c in self.children])
            clone.children = clone.children.append(child.set_parent(clone))
            return clone
        else:
            clone.children = OrderedSet([c.add_child(parent, child).set_parent(clone) for c in self.children])
        return clone

    def add_children(self, parent, children):
        clone = self.copy()
        if self == parent:
            clone.children = OrderedSet([c.set_parent(clone) for c in self.children])
            clone.children = clone.children.append_all([c.set_parent(clone) for c in children])
            return clone
        else:
            clone.children = OrderedSet([c.add_children(parent, children).set_parent(clone) for c in self.children])
        return clone

    def remove_child(self, child):
        parent = self.copy()
        for c in self.children:
            if c == child:
                continue
            else:
                parent.children = parent.children.append(c.remove_child(child).set_parent(parent))
        return parent

    def remove_children(self, children):
        parent = self.copy()
        for c in self.children:
            if c in set(children):
                continue
            else:
                parent.children.append(c.remove_children(children).set_parent(parent))
        return parent

    def find_node_with_domains(self, domains):
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

    def find_root_of_tree(self):
        if self.is_root():
            return self
        else:
            return self.parent.find_root_of_tree()
    def is_root(self):
        return False

    def create_value(self, domain: Domain, repr, hidden_keys: list[Domain], cardinality):
        return ColumnNode(domain, Val(), repr, hidden_keys, cardinality=cardinality)
    def check_if_value(self):
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

    def path_to_value(self, value):
        if self == value:
            if self.check_if_value():
                return [self]
            else:
                return None
        else:
            for child in self.children:
                suffix = child.path_to_value(value)
                if suffix is not None:
                    return [self] + suffix
            return None

    @classmethod
    def invert_path_internal(cls, path: list, table, namespace, prev_domains, hidden):
        if len(path) == 0:
            return []
        else:
            curr = path[0]
            inverted = invert_representation(curr.intermediate_representation)
            get_next_step = transform_step(namespace, table, curr.domains, prev_domains, [])
            inverted = invert_representation(curr.intermediate_representation)
            hidden_columns = []
            new_ir = []
            for step in inverted:
                next_step, cols = get_next_step(step)
                new_ir += [next_step]
                hidden_columns += cols
            curr.intermediate_representation = new_ir
            if len(path) == 1:
                inverted_tail = []
                inv_hidden = []
            else:
                curr.remove_child_node(path[1])
                path[1].parent = None
                inverted_tail, inv_hidden = DerivationNode.invert_path_internal(path[1:], table,
                                                                                namespace | set(curr.domains) | set(
                                                                                    prev_domains), curr.domains, hidden)
                inverted_tail[-1].add_child(curr)

            return inverted_tail + [curr], inv_hidden + [hidden_columns]

    @classmethod
    def invert_path(cls, path: list, table):
        start: DerivationNode = path[-1]
        end: DerivationNode = path[0]
        parent = end.parent

        # should be detached from tree
        assert parent is None

        curr: DerivationNode = path[-1]

        namespace = table.get_namespace()

        i = len(path) - 1
        new_path = []
        while i >= 0:
            prev_domains = []
            prev = None
            if i > 0:
                prev = path[i - 1]
                prev.remove_child_node(curr)
                curr.parent = None
                curr.add_child(prev)
                prev_domains = prev.domains

            inverted = invert_representation(curr.intermediate_representation)
            hidden_columns = []
            new_ir = []
            namespace |= set([d.name for d in curr.domains + prev_domains])
            get_next_step = transform_step(namespace, table, curr.domains, prev_domains, [])
            for step in inverted:
                next_step, cols = get_next_step(step)
                new_ir += [next_step]
                hidden_columns += cols
            curr.intermediate_representation = new_ir
            curr.hidden_keys = OrderedSet(hidden_columns)
            if curr.is_key_column():
                curr.to_value_column(Cardinality.MANY_TO_MANY)
            new_path += [curr]
            curr = prev
            i -= 1
        return new_path

    def remove_value(self, node):
        assert isinstance(node, ColumnNode) and isinstance(node.column_type, Val)
        if self == node:
            if len(self.children) == 0:
                return None
            else:
                intermediate_node = self.to_intermediate_node()
                intermediate_node.cardinality = self.cardinality
                return intermediate_node
        else:
            new_node = self.copy()
            new_children = []
            for child in self.children:
                new_child = child.remove_value(node)
                if new_child is not None:
                    new_children += [new_child]
            new_node.children = OrderedSet([])
            new_node.parent = None
            return new_node

    def to_intermediate_representation(self):
        ir = self.intermediate_representation.copy()  # push a frame onto the stack
        for i, child in enumerate(self.children):
            if i == 0:
                ir += [Call()] + child.to_intermediate_representation()
            else:
                ir += [Reset()] + child.to_intermediate_representation() + [Merge()]  # outer merge
        if len(self.children) > 0:
            ir += [Return()]  # right merge
        return ir

    def hide(self, column):
        idx = find_index(column, self.domains)
        if idx >= 0:
            new_node = self.copy()
            new_node.domains = new_node.domains[:idx] + new_node.domains[idx + 1:]
            new_node.children = OrderedSet([])
            for child in new_node.children:
                child.hidden_keys.append(column)
                child.parent = self
            return self
        else:
            children = []
            for child in self.children:
                children += [child.hide(col)]

            indices = {}
            old_children = np.array(self.children)

            for i, child in enumerate(children):
                assert len(child.domains) > 0
                doms = tuple(child.domains)
                if doms not in indices:
                    indices[doms] = []
                indices[doms] += [i]

            new_children = []
            for (i, child) in enumerate(old_children):
                to_union = old_children[indices[tuple(child.domains)]]
                new_child: DerivationNode = to_union[0]
                for sibling in to_union[1:]:
                    new_child.add_children(sibling.children)
                new_children += [new_child]

            self.remove_children([c.domains for c in self.children])
            self.add_children(new_children)
            return self

    def show(self, column):
        without_col = []
        without_col_idxs = {}
        with_col = []
        with_col_idxs = {}
        for child in self.children:
            if col not in child.hidden_keys.item_set:
                new = child.show(col)
                for new_child in new:
                    idx = find_index(col, new_child.domains)
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
                        base.add_nodes_as_children(new_child.children)
            else:
                child.hidden_keys.remove(col)
                new_child = child
                array = with_col
                idxs = with_col_idxs
                if new_child not in idxs:
                    idxs[new_child] = len(array)
                    array += [new_child]
                else:
                    base = array[idxs[new_child]]
                    base.add_nodes_as_children(new_child.children)

        if len(with_col) == 0:
            self.children = OrderedSet([])
            self.add_children(without_col)
            return [self]
        else:
            self.children = OrderedSet([])
            with_hk = self.copy()
            with_hk.children = OrderedSet([])
            without_hk = self.copy()
            without_hk.children = OrderedSet([])

            if with_hk.is_key_column():
                with_hk = with_hk.to_derivation_node()

            with_hk.domains += [column]

            with_hk.add_children(with_col)
            without_hk.add_children(without_col)

            return [with_hk, without_hk]

    def find_node_for_column(self, column):
        assert isinstance(column, ColumnNode)
        if self.domains == column.domains:
            return self
        else:
            for child in self.children:
                found = child.find_node_for_column(column)
                if found is not None:
                    return found

    def find_column_with_name(self, name: str):
        if isinstance(self, ColumnNode):
            if self.domains[0].name == name:
                return self
        for child in self.children:
            found = child.find_column_with_name(name)
            if found is not None:
                return found

    def get_hidden_keys_for_val(self) -> OrderedSet:
        return self.parent.get_hidden_keys_for_val().union(self.hidden_keys)

    def is_key_column(self):
        return False

    def is_val_column(self):
        return False

    def find_values(self):
        res = []
        if self.is_val_column():
            res += [self]
        for child in self.children:
            res += child.find_values()
        return res

    def find_keys(self):
        res = []
        if self.is_key_column():
            res += [self]
        for child in self.children:
            res += child.find_keys()
        return res

    def find_strong_keys(self):
        if self.parent.is_root():
            return self.domains
        else:
            return self.parent.find_strong_keys()

    def to_derivation_node(self):
        copy = DerivationNode(self.domains, self.intermediate_representation, self.hidden_keys)
        return copy.add_children(self.children)

    def to_key_column(self):
        copy = ColumnNode(self.domains[0], Key(), self.intermediate_representation, self.hidden_keys)
        return copy.add_children(self.children)

    def to_intermediate_node(self):
        intermediate = IntermediateNode(self.domains, self.intermediate_representation, self.hidden_keys)
        return intermediate.add_children(self.children)


    def __hash__(self):
        return hash((tuple([c.name for c in self.domains])))

    def __eq__(self, other):
        if isinstance(other, DerivationNode):
            return self.domains == other.domains
        raise NotImplemented()

    def __repr__(self):
        child_repr = ""
        for child in self.children:
            child_repr += '\n' + '\t' + child.__repr__().replace('\n', '\n\t')
        return f"{self.domains} // hidden: {self.hidden_keys.item_list}" + child_repr

    def __str__(self):
        return self.__repr__()

    def copy(self):
        copy = DerivationNode(self.domains, self.intermediate_representation, self.hidden_keys)
        return copy


class RootNode(DerivationNode):

    def __init__(self, keys: list[Domain]):
        super().__init__(keys, [Get(keys)])
        start_node = SchemaNode.product([k.node for k in keys])

        stt = StartTraversal(keys)
        prj = Project(start_node, None, [])
        ent = EndTraversal([], [])
        unit_node = DerivationNode([], [stt, prj, ent], [])
        self.children = self.children.append(unit_node)

        for i, key in enumerate(keys):
            stt = StartTraversal(keys)
            prj = Project(start_node, key.node, [i])
            ent = EndTraversal(keys, [key])
            key_node = ColumnNode(key, Key(), [stt, prj, ent], parent=self)
            self.children = self.children.append(key_node)


    def is_root(self):
        return True

    def forget(self, column):
        assert column.is_val_column()
        return self.remove_value(column)

    def hide(self, column):
        assert column.is_key_column()
        idx = find_index(column.get_domain(), self.domains)
        children = []
        for child in self.children:
            children += [child.hide(column.get_domain())]
        indices = {}
        new_children = []
        for child in children:
            if len(child.domains) == 0:
                child = child.to_derivation_node()
            if len(child.domains) == 1:
                child = child.to_key_column()
            if child not in indices:
                indices[child] = len(new_children)
                new_children += [child]
            else:
                new_children[indices[child]] = new_children[indices[child]].add_children(child.children)
        return RootNode(self.domains[:idx] + self.domains[idx + 1:]).add_children(new_children)

    def show(self, column: Domain):
        new_children = []
        new_children_indices = {}
        for child in self.children:
            new = child.show(column)
            for new_child in new:
                if new_child not in new_children_indices:
                    new_children_indices[new_child] = len(new_children)
                    new_children += [new_child]
                else:
                    base = new_children[new_children_indices[new_child]]
                    new_children[new_children_indices[new_child]] = base.add_children(new_child.children)
        new_root = RootNode(self.domains + [column]).add_children(new_children)
        return new_root

    def infer(self, value: Domain, strong_keys: list[Domain], old_hids: list[Domain], new_hids: list[Domain],
              intermediates: list, cardinality, repr: list[RepresentationStep]):
        assert self.parent is None
        to_prepend = []
        filtered = set([i.domains[0] for i in intermediates]).difference(set(strong_keys))
        new_root = self.insert_key(strong_keys)
        key_node = new_root.find_node_with_domains(strong_keys)
        val_node = self.create_value(value, repr, new_hids, cardinality)
        if len(filtered) > 0:
            for i, intermediate in enumerate(intermediates):
                intermediate_steps = intermediate.intermediate_representation
                if i == 0:
                    to_prepend += intermediate_steps
                else:
                    to_prepend += [Reset()] + intermediate_steps + [Merge()]
            intermediate_node = new_root.find_node_with_domains([i.domains[0] for i in intermediates])
            if intermediate_node is None:
                intermediate_cardinality = functools.reduce(compose_cardinality, [i.cardinality for i in intermediates])
                intermediate_node = IntermediateNode([i.domains[0] for i in intermediates], to_prepend, old_hids,
                                                     intermediate_cardinality)
                new_root = new_root.add_child(key_node, intermediate_node)
            new_root = new_root.add_child(intermediate_node, val_node)
        else:
            new_root = new_root.add_child(key_node, val_node)
        return new_root

    def compose(self, new_keys: list[Domain], old_key: Domain, hidden_keys, repr, cardinality):
        old_idx = find_index(old_key, self.domains)
        new_root = DerivationNode.create_root(self.domains[:old_idx] + new_keys + self.domains[old_idx+1:])
        for child in self.children:
            if child.domains == [old_key]:
                if len(child.children) == 0:
                    continue
                else:
                    new_root = new_root.insert_key(new_keys)
                    new_keys_node = new_root.find_node_with_domains(new_keys)
                    intermediate = IntermediateNode([old_key], repr, hidden_keys, None, [], cardinality)
                    new_root = new_root.add_child(new_keys_node, intermediate).add_children(intermediate, child.children)
            else:
                domains = child.domains
                idx = find_index(old_key, domains)

                if idx >= 0:
                    new_domains = domains[:idx] + new_keys + domains[idx + 1:]
                    new_root = new_root.insert_key(new_domains)
                    new_key_node = new_root.find_node_with_domains(new_domains)
                    intermediate = IntermediateNode(domains, repr, child.hidden_keys + hidden_keys)
                    new_root = new_root.add_child(new_key_node, intermediate).add_children(intermediate, child.children)

                else:
                    new_root = new_root.insert_key(child.domains)
                    key_node = new_root.find_node_with_domains(child.domains)
                    new_root = new_root.add_children(key_node, child.children)

        return new_root

    def is_node_in_tree(self, node):
        return functools.reduce(operator.or_, [c.is_node_in_subtree(node) for c in self.children], False)

    def get_values(self):
        return self.find_values()

    def get_keys(self):
        return self.find_keys()

    def get_keys_and_values(self):
        keys = self.get_keys()
        values = self.get_values()
        return keys + values

    def find_hidden(self, name: str):
        hidden_keys = self.get_all_hidden_keys_in_subtree()
        idx = find_index(name, [d.name for d in hidden_keys])
        return hidden_keys[idx]

    def get_hidden(self):
        return self.get_all_hidden_keys_in_subtree().to_list()

    def get_hidden_keys_for_val(self):
        return OrderedSet([])

    def insert_key(self, domains: list[Domain]):
        assert is_sublist(domains, self.domains)
        start_node = SchemaNode.product([d.node for d in self.domains])
        is_exact_match = np.sum(np.array([child.domains == domains for child in self.children]))

        if is_exact_match == 1:
            return self
        else:
            assert is_exact_match == 0
            stt = StartTraversal(self.domains)
            indices = [find_index(d, self.domains) for d in domains]
            prj = Project(start_node, SchemaNode.product([d.node for d in domains]), indices)
            ent = EndTraversal(domains, domains)
            node = DerivationNode(domains, [stt, prj, ent], [])
            new_root = self.copy()
            new_root.children = OrderedSet([c.set_parent(new_root) for c in self.children])
            new_root.children = new_root.children.append(node.set_parent(new_root))
            return new_root

    def add_child(self, parent, child):
        clone = self.copy()
        clone.children = OrderedSet([c.add_child(parent, child).set_parent(clone) for c in self.children])
        return clone

    def add_children(self, parent, children):
        clone = self.copy()
        clone.children = OrderedSet([c.add_children(parent, children).set_parent(clone) for c in self.children])
        return clone

    def remove_child(self, child):
        clone = self.copy()
        clone.children = OrderedSet([c.remove_child(child).set_parent(clone) for c in self.children])
        return clone

    def remove_children(self, children):
        clone = self.copy()
        clone.children = OrderedSet([c.remove_children(children).set_parent(clone) for c in self.children])
        return clone

    def copy(self):
        return RootNode(self.domains)

class ColumnNode(DerivationNode):
    def __init__(self, column: Domain, column_type: ColumnType,
                 intermediate_representation: list[RepresentationStep],
                 hidden_keys: list[Domain] | OrderedSet = None, parent=None, cardinality=None):
        super().__init__([column], intermediate_representation, hidden_keys=hidden_keys, parent=parent)
        self.name = column.name
        self.column_type = column_type
        self.cardinality = cardinality

    def is_key_column(self):
        return self.column_type.is_key_column()

    def is_val_column(self):
        return self.column_type.is_val_column()

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

    def duplicate(self):
        node = ColumnNode(self.get_domain(), self.column_type, self.intermediate_representation)
        node.hidden_keys = self.hidden_keys
        node.parent = self.parent
        node.children = self.children
        node.cardinality = self.cardinality
        return node

    def copy(self):
        copy = ColumnNode(self.domains[0], self.column_type, self.intermediate_representation.copy(),
                          self.hidden_keys, cardinality=self.cardinality)
        return copy

    def to_value_column(self, cardinality):
        self.column_type = Val()
        self.cardinality = cardinality


class IntermediateNode(DerivationNode):
    def __init__(self, domains: list[Domain], intermediate_representation: list[RepresentationStep],
                 hidden_keys: list[Domain] | OrderedSet = None, parent=None,
                 children: list | OrderedSet = None,
                 cardinality: Cardinality = None):
        super().__init__(domains, intermediate_representation, hidden_keys, parent, children)
        self.cardinality = cardinality

    def to_intermediate_representation(self):
        assert len(self.children) > 0
        assert self.parent is not None
        root = self.find_root_of_tree()
        keys = set([n.domains[0] for n in root.get_keys()])
        vals = set([n.domains[0] for n in self.find_values()])
        intermediates = [c for c in self.domains if c not in keys | vals]
        ir = super().to_intermediate_representation()
        return ir[:-1] + [Drop(intermediates)] + [Return()]

    def copy(self):
        copy = IntermediateNode(self.domains, self.intermediate_representation.copy(),
                                self.hidden_keys, cardinality=self.cardinality)
        return copy
