import abc
import functools
import operator

import numpy as np

from helpers.compose_cardinality import compose_cardinality
from schema import SchemaNode, is_sublist, Cardinality
from schema.helpers.find_index import find_index
from schema.helpers.invert_representation import invert_representation
from tables.helpers.transform_step import transform_step
from tables.internal_representation import RepresentationStep, StartTraversal, Merge, Return, Call, Reset, \
    Drop, Project, EndTraversal, Get
from tables.domain import Domain


class OrderedSet:
    def __init__(self, items: list = None):
        self.item_list = []
        self.item_set = frozenset([])
        if items is None:
            self.item_set = frozenset()
            self.item_list = []
        else:
            for item in items:
                if item not in self.item_set:
                    self.item_set |= {item}
                    self.item_list += [item]

    def append(self, item):
        if item not in self.item_set:
            self.item_set |= {item}
            self.item_list += [item]

    def append_all(self, items):
        for item in items:
            self.append(item)

    def find_item(self, item):
        idx = find_index(item, self.item_list)
        if idx >= 0:
            return self.item_list[idx]

    def find_item_with_idx(self, item):
        idx = find_index(item, self.item_list)
        if idx >= 0:
            return idx, self.item_list[idx]

    def remove(self, item):
        idx, to_remove = self.find_item_with_idx(item)
        if to_remove is not None:
            self.item_set -= {to_remove}
            self.item_list = self.item_list[:idx] + self.item_list[idx + 1:]

    def remove_all(self, items):
        for item in items:
            self.remove(item)

    def union(self, other):
        if isinstance(other, OrderedSet):
            self.append_all(other.item_list)
        else:
            raise NotImplemented()

    def __len__(self):
        return len(self.item_list)

    def __iter__(self):
        return self.item_list.__iter__()

    def __getitem__(self, item):
        return self.item_list[item]


class DerivationNode:
    def __init__(self, domains: list[Domain], intermediate_representation: list[RepresentationStep],
                 hidden_keys: list[Domain] = None, parent=None,
                 children: list[tuple[SchemaNode, list[RepresentationStep], list[Domain]]] = None):
        self.domains = domains
        self.intermediate_representation = intermediate_representation
        self.on_completion = None
        self.parent = parent
        if children is None:
            self.children = OrderedSet()
        else:
            self.children = OrderedSet([DerivationNode(n, ir, hk, parent=self) for (n, ir, hk) in children])
        if hidden_keys is None:
            self.hidden_keys = OrderedSet()
        else:
            self.hidden_keys = OrderedSet(hidden_keys)

    @classmethod
    def create_root(cls, domains: list[Domain]):
        root = DerivationNode(domains, [Get(domains)], [])
        start_node = SchemaNode.product([d.node for d in domains])
        for i, key in enumerate(domains):
            stt = StartTraversal(domains)
            prj = Project(start_node, key.node, [i])
            ent = EndTraversal(domains, [key])
            key_node = ColumnNode(key, Key(), [stt, prj, ent])
            root.add_node_as_child(key_node)
        return root

    def find_hidden(self, name: str):
        assert self.parent is None
        idx = find_index(name, [d.name for d in self.get_hidden()])
        return self.get_hidden()[idx]

    def is_node_in_subtree(self, node):
        if self == node:
            return True
        if len(self.children) == 0:
            return False
        else:
            return functools.reduce(operator.or_, [c.is_node_in_subtree(node) for c in self.children])

    def is_node_in_tree(self, node):
        if self.parent is not None:
            return self.parent.is_node_in_tree(node)
        else:
            return self.is_node_in_subtree(node)

    def add_node_as_child(self, node):
        assert not self.is_node_in_tree(node)
        self.hidden_keys.union(OrderedSet(node.hidden_keys))
        node.parent = self
        self.children.append(node)
        return node

    def add_nodes_as_children(self, nodes):
        for node in nodes:
            self.add_node_as_child(node)

    def recompute_hidden_keys(self):
        new_hidden_keys = OrderedSet()
        for child in self.children:
            new_hidden_keys.union(child.hidden_keys)

    def remove_child(self, domains: list[Domain]):
        idx = find_index(domains, [c.domains for c in self.children])
        child = self.children[idx]
        self.children.remove(child)
        # TODO: Batch it together
        self.recompute_hidden_keys()
        child.parent = None
        return child

    def remove_children(self, children: list[list[Domain]]):
        for child in children:
            self.remove_child(child)

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
        return self.parent is None

    def insert_key(self, domains: list[Domain]):
        # insert at highest level
        # structure should be root <key <value, value, value>, key <value, value>>
        if not self.is_root():
            root = self.find_root_of_tree()
            return root.insert_key(domains)
        assert is_sublist(domains, self.domains)
        # if self.domains == domains:
        #     return self
        start_node = SchemaNode.product([d.node for d in self.domains])
        is_exact_match = [child.domains == domains for child in self.children]
        if len(domains) == 0:
            if np.sum(np.array(is_exact_match)) == 1:
                child, = np.array(self.children)[np.array(is_exact_match)]
                return child
            else:
                unit_node = DerivationNode([], [], [])
                self.add_node_as_child(unit_node)
        elif np.sum(np.array(is_exact_match)) == 1:
            child, = (np.array(self.children))[np.array(is_exact_match)]
            return child
        elif np.sum(np.array(is_exact_match)) == 0:
            stt = StartTraversal(self.domains)
            indices = [find_index(d, self.domains) for d in domains]
            prj = Project(start_node, SchemaNode.product([d.node for d in domains]), indices)
            ent = EndTraversal(domains, domains)
            node = DerivationNode(domains, [stt, prj, ent], [])
            new_child = self.add_node_as_child(node)
            return new_child
        else:
            assert False

    def insert_value(self, domain: Domain, strong_keys, repr, hidden_keys: list[Domain], cardinality):
        node = ColumnNode(domain, Val(), repr, hidden_keys, cardinality=cardinality)
        child = self.add_node_as_child(node)
        return child


    def check_if_value(self):
        if self.is_val_column():
            return True
        domains = self.domains
        for domain in self.domains:
            idx = find_index([domain], [c.domains for c in self.children.item_list])
            if idx < 0:
                return False
            child = self.children[idx]
            if not child.is_val_column():
                return False
        return True

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

    def remove_child_node(self, node):
        self.children.remove(node)


    @classmethod
    def invert_path(cls, path: list, table):
        namespace = table.get_namespace()
        start: DerivationNode = path[-1]
        end: DerivationNode = path[0]
        parent = end.parent

        # should be detached from tree
        assert parent is None

        curr: DerivationNode = path[-1]

        get_next_step = transform_step(namespace, table, start, end, [])

        i = len(path)-1
        while i > 0:
            prev = curr.parent
            prev.remove_child_node(curr)
            curr.add_node_as_child(prev)
            inverted = invert_representation(curr.intermediate_representation)
            hidden_columns = []
            new_ir = []
            for step in inverted:
                next_step, cols = get_next_step(step)
                new_ir += [next_step]
                hidden_columns += cols
            curr.intermediate_representation = new_ir
            if len(hidden_columns) > 0:
                curr.propagate_hidden_keys(hidden_columns)
            curr = prev
        return path[::-1]

    def propagate_hidden_keys(self, hidden_keys):
        to_append = hidden_keys
        domains = []
        if self.parent is not None:
            domains = self.parent.domains
        if is_sublist(self.domains, domains):
            to_append = [hk for hk in hidden_keys if hk not in set(self.parent.domains).difference(self.domains)]
        self.hidden_keys.append_all(to_append)
        for child in self.children:
            child.propagate_hidden_keys(to_append)

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
            new_children = []
            for child in self.children:
                new_child = child.remove_value(node)
                if new_child is not None:
                    new_children += [new_child]
            self.children = OrderedSet([])
            self.add_nodes_as_children(new_children)
            return self


    def set_parent(self, parent):
        self.parent = parent

    def set_intermediate_representation(self, new_representation):
        self.intermediate_representation = new_representation

    def to_intermediate_representation(self):
        ir = self.intermediate_representation.copy()  # push a frame onto the stack
        for i, child in enumerate(self.children):
            if i == 0:
                ir += [Call()] + child.to_intermediate_representation()
            else:
                ir += [Reset()] + child.to_intermediate_representation() + [Merge()]  # outer merge
        if len(self.children) > 0:
            ir += [Return()]  # right merge
        if self.parent is None:
            print(ir)
        return ir

    def infer(self, value: Domain, strong_keys: list[Domain], old_hids: list[Domain], new_hids: list[Domain],
              intermediates: list, cardinality, repr: list[RepresentationStep]):
        assert self.parent is None
        to_prepend = []
        for i, intermediate in enumerate(intermediates):
            intermediate_steps = intermediate.intermediate_representation
            if i == 0:
                to_prepend += intermediate_steps
            else:
                to_prepend += [Reset()] + intermediate_steps + [Merge()]
        key_node = self.insert_key(strong_keys)
        intermediate_node = key_node.find_node_with_domains([i.domains[0] for i in intermediates])
        if intermediate_node is None:
            intermediate_cardinality = functools.reduce(compose_cardinality, [i.cardinality for i in intermediates])
            intermediate_node = IntermediateNode([i.domains[0] for i in intermediates], to_prepend, old_hids, intermediate_cardinality)
            key_node.add_node_as_child(intermediate_node)
        intermediate_node.insert_value(value, strong_keys, repr, old_hids + new_hids, cardinality)

    def hide_internal(self, column):
        col = column
        idx = find_index(col, self.domains)
        if idx >= 0:
            self.domains = self.domains[:idx] + self.domains[idx+1:]
            self.hidden_keys.append(col)

        for child in self.children:
            child.hide_internal(col)

        to_remove = []
        unit_deriv = -1
        indices = {}
        old_children = np.array(self.children)

        for i, child in enumerate(self.children):
            if len(child.domains) == 0:
                if child.is_key_column():
                    to_remove += [i]
                else:
                    unit_deriv = i
                continue
            doms = tuple(child.domains)
            if doms not in indices:
                indices[doms] = []
            indices[doms] += [i]

        new_children = []
        for (i, child) in enumerate(old_children):
            if child is None or i == unit_deriv or i in set(to_remove):
                continue
            to_union = old_children[indices[tuple(child.domains)]]
            new_child: DerivationNode = to_union[0]
            for sibling in to_union[1:]:
                new_child.add_nodes_as_children(sibling.children)
            new_children += [new_child]

        if unit_deriv >= 0:
            root = self.find_root_of_tree()
            unit = root.insert_key([])
            unit.add_nodes_as_children(unit_deriv.children)
            self.remove_child([])

        self.remove_children([c.domains for c in self.children])
        self.add_nodes_as_children(new_children)

    def show_internal(self, column):
        col = column

        new_children = []
        new_children_indices = {}
        for child in self.children:
            new = child.show_internal(col)
            for new_child in new:
                if new_child not in new_children_indices:
                    new_children_indices[new_child] = len(new_children)
                    new_children += [new_child]
                else:
                    base = new_children[new_children_indices[new_child]]
                    base.add_nodes_as_children(new_child.children)

        idx = find_index(col, self.get_hidden())
        if idx < 0:
            self.children = OrderedSet([])
            self.add_nodes_as_children(new_children)
            return [self]
        else:
            with_hk = self.duplicate()
            without_hk = self.duplicate()

            with_hk.domains += [column]
            with_hk.hidden_keys.remove(col)

            for child in new_children:
                c_idx = find_index(col, child.domains)
                if c_idx >= 0:
                    with_hk.add_node_as_child(child)
                else:
                    without_hk.add_node_as_child(child)
            return [with_hk, without_hk]

    def forget(self, column):
        assert self.parent is None
        assert column.is_val_column()
        self.remove_value(column)

    def hide(self, column):
        assert self.parent is None
        assert column.is_key_column()
        self.hide_internal(column.get_domain())

    def show(self, column: Domain):
        assert self.parent is None
        new_children = []
        new_children_indices = {}
        is_new_key_child = False
        for child in self.children:
            new = child.show_internal(column)
            for new_child in new:
                if new_child not in new_children_indices:
                    if len(new_child.domains) == 1 and new_child.domains[0] == column:
                        is_new_key_child = True
                    new_children_indices[new_child] = len(new_children)
                    new_children += [new_child]
                else:
                    base = new_children[new_children_indices[new_child]]
                    base.add_nodes_as_children(new_child.children)
        self.domains += [column]
        self.hidden_keys.remove(column)
        self.children = OrderedSet([])
        self.add_nodes_as_children(new_children)
        if not (is_new_key_child):
            new_key = ColumnNode(column, Key(), [], [])
            self.add_node_as_child(new_key)


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

    def get_values(self):
        res = []
        if self.is_val_column():
            res += [self]
        for child in self.children:
            res += child.get_values()
        return res

    def get_keys(self):
        res = []
        if self.is_key_column():
            res += [self]
        for child in self.children:
            res += child.get_keys()
        return res

    def get_keys_and_values(self):
        assert self.parent is None
        keys = self.get_keys()
        values = self.get_values()
        return keys + values

    def get_hidden(self):
        return self.hidden_keys.item_list

    def is_key_column(self):
        return False

    def is_val_column(self):
        return False

    def find_strong_keys(self):
        if self.parent.parent is None:
            return self.parent.domains
        else:
            return self.parent.find_strong_keys()

    def to_value_column(self, cardinality):
        return ColumnNode(self.domains[0], Val(), self.intermediate_representation, self.hidden_keys.item_list, self.parent, cardinality)

    def to_key_column(self):
        return ColumnNode(self.domains[0], Key(), self.intermediate_representation, self.hidden_keys.item_list, self.parent)

    def duplicate(self):
        node = DerivationNode(self.domains, self.intermediate_representation)
        node.hidden_keys = self.hidden_keys
        node.parent = self.parent
        node.children = self.children
        return node

    def __hash__(self):
        if self.parent is None:
            return hash((tuple([c.name for c in self.domains]), tuple([c.name for c in self.hidden_keys])))
        return hash((tuple([c.name for c in self.domains]), tuple([c.name for c in self.hidden_keys]), self.parent))

    def __eq__(self, other):
        if isinstance(other, DerivationNode):
            if self.parent is None:
                return self.domains == other.domains and self.hidden_keys == other.hidden_keys
            else:
                return self.domains == other.domains and self.hidden_keys == other.hidden_keys and self.parent == other.parent
        raise NotImplemented()

    def __repr__(self):
        child_repr = ""
        for child in self.children:
            child_repr += '\n' + '\t' + child.__repr__().replace('\n', '\n\t')
        return f"{self.domains} // hidden: {self.hidden_keys.item_list}" + child_repr

    def __str__(self):
        return self.__repr__()

    def to_intermediate_node(self):
        intermediate = IntermediateNode(self.domains, self.intermediate_representation,
                                self.hidden_keys.item_list)
        intermediate.add_nodes_as_children(self.children)
        return intermediate



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
        return node.get_hidden()

    def get_derivation(self, node):
        return self.get_strong_keys(node), self.get_hidden_keys(node)


class ColumnNode(DerivationNode):
    def __init__(self, column: Domain, column_type: ColumnType,
                 intermediate_representation: list[RepresentationStep],
                 hidden_keys: list[Domain] = None, parent=None, cardinality=None):
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

class IntermediateNode(DerivationNode):
    def __init__(self, domains: list[Domain], intermediate_representation: list[RepresentationStep],
                 hidden_keys: list[Domain] = None, parent=None,
                 children: list[tuple[SchemaNode, list[RepresentationStep], list[Domain]]] = None,
                 cardinality: Cardinality = None):
        super().__init__(domains, intermediate_representation, hidden_keys, parent, children)
        self.cardinality = cardinality

    def to_intermediate_representation(self):
        assert len(self.children) > 0
        assert self.parent is not None
        root = self.find_root_of_tree()
        keys = set([n.domains[0] for n in root.get_keys()])
        intermediates = [c for c in self.domains if c not in keys]
        ir = super().to_intermediate_representation()
        return ir[:-1] + [Drop(intermediates)] + [Return()]

    def duplicate(self):
        node = IntermediateNode(self.domains, self.intermediate_representation)
        node.hidden_keys = self.hidden_keys
        node.parent = self.parent
        node.children = self.children
        node.cardinality = self.cardinality
        return node
