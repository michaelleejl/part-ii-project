import functools
import operator

import numpy as np

from helpers.compose_cardinality import compose_cardinality
from schema import SchemaNode, is_sublist, Cardinality
from schema.helpers.find_index import find_index
from schema.helpers.invert_representation import invert_representation
from tables.derivation.column_type import ColumnType, Key, Val
from tables.derivation.ordered_set import OrderedSet
from tables.helpers.rename_column_in_representation import rename_column_in_representation
from tables.helpers.transform_step import transform_step
from tables.internal_representation import *
from tables.domain import Domain


def create_value(domain: Domain, repr, hidden_keys: list[Domain], cardinality):
    return ColumnNode(domain, Val(), repr, hidden_keys, cardinality=cardinality)


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
        clone = self.copy()
        clone.parent = parent
        clone.children = OrderedSet([c.set_parent(clone) for c in self.children])
        return clone

    def add_child(self, parent, child):
        clone = self.copy()
        if self == parent:
            clone.children = OrderedSet([c.set_parent(clone) for c in self.children])
            clone.children = clone.children.append(child.set_parent(clone))
            return clone
        else:
            clone.children = OrderedSet([c.set_parent(clone).add_child(parent, child) for c in self.children])
        return clone

    def add_children(self, parent, children):
        clone = self.copy()
        if self == parent:
            clone.children = OrderedSet([c.set_parent(clone) for c in self.children])
            clone.children = clone.children.append_all([c.set_parent(clone) for c in children])
            return clone
        else:
            clone.children = OrderedSet([c.set_parent(clone).add_children(parent, children) for c in self.children])
        return clone

    def remove_child(self, child):
        parent = self.copy()
        for c in self.children:
            if c == child:
                continue
            else:
                parent.children = parent.children.append(c.set_parent(parent).remove_child(child))
        return parent

    def remove_children(self, children):
        parent = self.copy()
        for c in self.children:
            if c in set(children):
                continue
            else:
                parent.children.append(c.set_parent(parent).remove_children(children))
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

    def intermediate_representation_for_path(self, path, keys, hids):
        intermediate_representation = []
        for node in path:
            intermediate_representation += node.intermediate_representation
        ents = [step for step in intermediate_representation if isinstance(step, EndTraversal)]
        to_add = functools.reduce(lambda x, y: x.union(y), [step.end_columns for step in ents[:-1]], set())
        to_remove = set(ents[-1].end_columns)
        return intermediate_representation + [Drop(list(to_add.difference(to_remove).difference(set(keys).union(set(hids)))))]

    @classmethod
    def invert_path(cls, path: list, table):
        start: DerivationNode = path[-1]
        end: DerivationNode = path[0]
        parent = end.parent

        assert parent is None

        namespace = table.get_namespace()

        curr: DerivationNode = path[-2].copy()

        start_inverted = invert_representation(start.intermediate_representation)
        new_start_ir = []
        start_hidden_cols = []
        namespace |= set([d.name for d in start.domains + curr.domains])
        get_next_step = transform_step(namespace, table, start.domains, curr.domains, [])
        for step in start_inverted:
            next_step, cols = get_next_step(step)
            new_start_ir += [next_step]
            start_hidden_cols += cols
        intermediate_representation = new_start_ir
        hidden_keys = start_hidden_cols

        i = len(path) - 2
        start = start.set_parent(None)
        new_path = start
        parent = start

        path_representation = [new_start_ir]

        while i >= 0:
            child = None
            child_domains = []
            if i > 0:
                child = path[i-1]
                child_domains = child.domains
            curr.children = OrderedSet([c for c in curr.children if c != parent])

            inverted = invert_representation(curr.intermediate_representation)
            hidden_columns = []
            new_ir = []
            namespace |= set([d.name for d in curr.domains + child_domains])
            get_next_step = transform_step(namespace, table, curr.domains, child_domains,[])
            for step in inverted:
                next_step, cols = get_next_step(step)
                new_ir += [next_step]
                hidden_columns += cols

            curr.intermediate_representation = intermediate_representation
            path_representation = [intermediate_representation] + path_representation
            curr.hidden_keys = OrderedSet(hidden_keys)

            intermediate_representation = new_ir
            hidden_keys = hidden_columns

            if curr.is_key_column():
                curr = curr.to_value_column(Cardinality.MANY_TO_MANY)

            new_path = new_path.add_child(parent, curr)
            parent = curr
            curr = child
            i -= 1
        assert len(new_path.children) == 1
        return new_path.children[0], path_representation

    def remove_value(self, node):
        assert isinstance(node, ColumnNode) and isinstance(node.column_type, Val)
        if self == node:
            if len(self.children) == 0:
                return None
            else:
                intermediate_node = self.to_intermediate_node()
                intermediate_node.cardinality = self.cardinality
                intermediate_node.children = OrderedSet([c.set_parent(intermediate_node) for c in self.children])
                return intermediate_node
        else:
            new_node = self.copy()
            new_children = []
            for child in self.children:
                new_child = child.set_parent(new_node).remove_value(node)
                if new_child is not None:
                    new_children += [new_child]
            new_node.children = OrderedSet(new_children)
            if len(new_node.children) == 0 and new_node.is_intermediate_node():
                return None
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
            new_columns = new_node.domains[:idx] + new_node.domains[idx + 1:]
            new_node.domains = new_columns

            if len(new_node.domains) == 0 and new_node.is_key_column() or new_node.is_val_column():
                new_node = new_node.to_derivation_node()

            new_node.children = OrderedSet([])

            for child in self.children:
                new_child = child.hide(column)
                new_child.parent = new_node
                old_columns = self.domains
                intermediate_node = new_node.find_node_with_domains(old_columns)
                if len(new_columns) == 0:
                    intermediate_representation = [Pop(), Get(old_columns)]
                else:
                    start_node = SchemaNode.product([d.node for d in new_columns])
                    end_node = SchemaNode.product([d.node for d in old_columns])
                    indices = [i for i in range(len(old_columns)) if i != idx]
                    intermediate_representation = [StartTraversal(new_columns),
                                                   Expand(start_node, end_node, indices, [column], old_columns),
                                                   EndTraversal(new_columns, old_columns)]
                if intermediate_node is None:
                    intermediate_node = DerivationNode(old_columns, intermediate_representation, [column])
                    new_node = new_node.add_child(new_node, intermediate_node)
                new_node = new_node.add_child(intermediate_node, new_child)
        else:
            new_node = self.copy()
            for child in self.children:
                new_child = child.hide(column)
                idx = find_index(new_child, new_node.children.to_list())
                if idx >= 0:
                    child_node = new_node.children[idx]
                    new_node = new_node.add_children(child_node, new_child.children)
                else:
                    new_node = new_node.add_child(new_node, new_child)
        return new_node

    def show(self, column):
        without_col = []
        without_col_idxs = {}
        with_col = []
        with_col_idxs = {}
        for child in self.children:
            if column not in child.hidden_keys:
                new_children = child.show(column)
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
                        array[idxs[new_child]] = base.add_children(base, new_child.children)
            else:
                new_child = child.copy()
                new_child.hidden_keys = new_child.hidden_keys.remove(column)
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
            new_node.children = OrderedSet([c.set_parent(new_node) for c in without_col])
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

    def equate(self, key1, key2):
        idx1 = find_index(key1, self.domains)
        idx2 = find_index(key2, self.domains)

        new_node = self.copy()

        if idx2 >= 0:
            if idx1 >= 0:
                new_node.domains = self.domains[:idx2] + self.domains[idx2+1:]
                start_node = SchemaNode.product([d.node for d in new_node.domains])
                end_node = key1.node
                ir = [StartTraversal(new_node.domains),
                      Project(start_node, end_node, [idx1]),
                      EndTraversal(new_node.domains, [key2])]
                intermediate = IntermediateNode(self.domains, ir, parent=new_node)
                intermediate = intermediate.add_children(intermediate, self.children)
                new_node.children = OrderedSet([intermediate])
            else:
                new_node.domains = self.domains[:idx2] + [key1] + self.domains[idx2+1:]
                start_node = SchemaNode.product([d.node for d in new_node.domains])
                end_node = key1.node
                ir = [StartTraversal(new_node.domains),
                      Project(start_node, end_node, [idx2]),
                      EndTraversal(new_node.domains, [key2])]
                intermediate = IntermediateNode(self.domains, ir, parent=new_node)
                intermediate = intermediate.add_children(intermediate, self.children)
                new_node.children = OrderedSet([intermediate])
                return new_node
        else:
            new_node.children = OrderedSet([c.set_parent(new_node) for c in self.children])

        return new_node

    def rename(self, old_name: str, new_name: str):
        idx = find_index(old_name, [d.name for d in self.domains])
        if idx >= 0:
            old_domain = self.domains[idx]
            new_domain = Domain(new_name, old_domain.node)
            new_domains = self.domains[:idx] + [new_domain] + self.domains[idx + 1:]
        else:
            new_domains = self.domains
        new_node = self.copy()
        new_node.domains = new_domains.copy()
        new_node.intermediate_representation = rename_column_in_representation(self.intermediate_representation, old_name, new_name)
        new_node.children = OrderedSet([c.set_parent(new_node).rename(old_name, new_name) for c in self.children])
        return new_node

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

    def is_intermediate_node(self):
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
        copy.parent = self.parent
        copy = copy.add_children(copy, self.children)
        return copy

    def to_key_column(self):
        copy = ColumnNode(self.domains[0], Key(), self.intermediate_representation, self.hidden_keys)
        copy.parent = self.parent
        copy = copy.add_children(copy, self.children)
        return copy

    def to_value_column(self, cardinality):
        copy = ColumnNode(self.domains[0], Val(), self.intermediate_representation, self.hidden_keys, cardinality)
        copy.parent = self.parent
        copy = copy.add_children(copy, self.children)
        return copy

    def to_intermediate_node(self):
        intermediate = IntermediateNode(self.domains, self.intermediate_representation, self.hidden_keys)
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
            child_repr += '\n' + '\t' + child.__repr__().replace('\n', '\n\t')
        return f"{self.domains} // hidden: {self.hidden_keys.item_list}" + child_repr

    def __str__(self):
        return self.__repr__()

    def copy(self):
        copy = DerivationNode(self.domains, self.intermediate_representation, self.hidden_keys)
        copy.parent = self.parent
        copy.children = self.children
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

    # inner product
    def equate(self, key1, key2):
        idx = find_index(key2, self.domains)
        new_keys = self.domains[:idx] + self.domains[idx+1:]
        new_root = RootNode(new_keys)
        for child in self.children:
            new_child = child.equate(key1, key2)
            new_root = new_root.insert_key(new_child.domains)
            key_node = new_root.find_node_with_domains(new_child.domains)
            new_root = new_root.add_children(key_node, new_child.children)
        return new_root


    def forget(self, column):
        assert column.is_val_column()
        new_root = self.copy()
        new_root.children = OrderedSet([c.remove_value(column).set_parent(new_root) for c in self.children])
        return new_root

    def hide(self, column):
        assert column.is_key_column()
        idx = find_index(column.get_domain(), self.domains)
        new_root = RootNode(self.domains[:idx] + self.domains[idx + 1:])
        for child in self.children:
            new_child = child.hide(column.get_domain())
            new_root = new_root.insert_key(new_child.domains)
            key_node = new_root.find_node_with_domains(new_child.domains)
            new_root = new_root.add_children(key_node, new_child.children)
        return new_root

    def show(self, column: Domain):
        new_root = RootNode(self.domains + [column])
        for child in self.children:
            new_children = child.show(column)
            for new_child in new_children:
                new_root = new_root.insert_key(new_child.domains)
                key_node = new_root.find_node_with_domains(new_child.domains)
                new_root = new_root.add_children(key_node, new_child.children)
        return new_root

    def infer(self, value: Domain, strong_keys: list[Domain], old_hids: list[Domain], new_hids: list[Domain],
              intermediates: list, cardinality, repr: list[RepresentationStep]):
        assert self.parent is None
        to_prepend = []
        filtered = set([i.domains[0] for i in intermediates]).difference(set(strong_keys))
        new_root = self.insert_key(strong_keys)
        key_node = new_root.find_node_with_domains(strong_keys)
        val_node = create_value(value, repr, new_hids, cardinality)
        j = 0
        if len(filtered) > 0:
            for i, intermediate in enumerate(intermediates):
                if intermediate.is_val_column():
                    i_keys = intermediate.get_strong_keys()
                    hid_keys = intermediate.get_hidden_keys_for_val()
                    key_node = self.find_node_with_domains(i_keys)
                    path = key_node.path_to_value(intermediate)
                    intermediate_steps = self.intermediate_representation_for_path(path, strong_keys, hid_keys)
                    j+=1
                    if j == 0:
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
        new_root = DerivationNode.create_root(self.domains[:old_idx] + new_keys + self.domains[old_idx + 1:])
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
                    intermediate = IntermediateNode([old_key], repr, hidden_keys, None, [], cardinality)
                    new_root = new_root.add_child(new_keys_node, intermediate).add_children(intermediate,
                                                                                            child.children)
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

    def rename(self, old_name: str, new_name: str):
        idx = find_index(old_name, [d.name for d in self.domains])
        if idx >= 0:
            old_domain = self.domains[idx]
            new_domain = Domain(new_name, old_domain.node)
            new_domains = self.domains[:idx] + [new_domain] + self.domains[idx+1:]
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
        clone.children = OrderedSet([c.set_parent(clone).add_child(parent, child) for c in self.children])
        return clone

    def add_children(self, parent, children):
        clone = self.copy()
        clone.children = OrderedSet([c.set_parent(clone).add_children(parent, children) for c in self.children])
        return clone

    def remove_child(self, child):
        clone = self.copy()
        clone.children = OrderedSet([c.set_parent(clone).remove_child(child) for c in self.children])
        return clone

    def remove_children(self, children):
        clone = self.copy()
        clone.children = OrderedSet([c.set_parent(clone).remove_children(children) for c in self.children])
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

    def copy(self):
        copy = ColumnNode(self.domains[0], self.column_type, self.intermediate_representation.copy(),
                          self.hidden_keys, cardinality=self.cardinality)
        copy.parent = self.parent
        copy.children = self.children
        return copy


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
        intermediates = self.get_intermediates()
        ir = super().to_intermediate_representation()
        return ir[:-1] + [Drop(intermediates)] + [Return()]

    def get_intermediates(self):
        root = self.find_root_of_tree()
        keys = set([n.domains[0] for n in root.get_keys()])
        vals = set([n.domains[0] for n in self.find_values()])
        hids = set(self.get_hidden_keys_for_val())
        intermediates = [c for c in self.domains if c not in keys | vals | hids]
        return intermediates

    def is_intermediate_node(self):
        return True

    def copy(self):
        copy = IntermediateNode(self.domains, self.intermediate_representation.copy(),
                                self.hidden_keys, cardinality=self.cardinality)
        copy.parent = self.parent
        copy.children = self.children
        return copy
