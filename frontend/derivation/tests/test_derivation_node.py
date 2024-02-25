import expecttest
from frontend.derivation.derivation_node import (
    DerivationNode,
    ColumnNode,
    intermediate_representation_for_path,
    invert_derivation_path,
    set_and_name_hidden_keys_along_path,
    find_splice_point,
)
from frontend.derivation.ordered_set import OrderedSet
from frontend.derivation.exceptions import *
from frontend.domain import Domain
from frontend.mapping import Mapping
from frontend.tables.column_type import Val, Key, HiddenKey
from schema.cardinality import Cardinality
from schema.edge import SchemaEdge
from schema.node import AtomicNode, SchemaNode
from representation.representation import (
    StartTraversal,
    Traverse,
    EndTraversal,
    Get,
    Project,
)


class TestDerivationNode(expecttest.TestCase):

    def test_create_root(self):
        u = AtomicNode("u")
        v = AtomicNode("v")
        w = AtomicNode("w")
        a = Domain("a", u)
        b = Domain("b", v)
        c = Domain("c", w)

        root = DerivationNode.create_root([a, b, c])

        self.assertExpectedInline(
            str(root),
            """\
[a, b, c] // hidden: []
	[] // hidden: []
	[a] // hidden: []
	[b] // hidden: []
	[c] // hidden: []""",
        )

    def test_create_root_raisesExceptionIfDomainsNotUnique(self):
        u = AtomicNode("u")
        v = AtomicNode("v")
        w = AtomicNode("w")
        a = Domain("a", u)
        b = Domain("b", v)
        c = Domain("a", w)

        self.assertExpectedRaisesInline(
            KeysMustBeUniqueException,
            lambda: DerivationNode.create_root([a, b, c]),
            """Keys [a, b, a] are not unique""",
        )

    def test_is_node_in_tree_returnsTrueIfNodeInTree(self):
        u = AtomicNode("u")
        v = AtomicNode("v")
        w = AtomicNode("w")
        a = Domain("a", u)
        b = Domain("b", v)
        c = Domain("c", w)

        root = DerivationNode.create_root([a, b, c])
        node_a = root.children[1]

        self.assertTrue(root.is_node_in_tree(node_a))

    def test_is_node_in_tree_returnsFalseIfNodeNotInTree(self):
        u = AtomicNode("u")
        v = AtomicNode("v")
        w = AtomicNode("w")
        a = Domain("a", u)
        b = Domain("b", v)
        c = Domain("c", w)

        z = Domain("z", w)
        not_in_tree = DerivationNode([z], [])
        root = DerivationNode.create_root([a, b, c])

        self.assertFalse(root.is_node_in_tree(not_in_tree))

    def test_set_parent_successfullySetsParentOfNode(self):
        u = AtomicNode("u")
        c_domain = Domain("c", u)
        c_node = DerivationNode([c_domain], [])
        p_domain = Domain("p", u)
        p_node = DerivationNode([p_domain], [])
        new_c = c_node.set_parent(p_node)
        self.assertExpectedInline(str(new_c.parent), """[p] // hidden: []""")

    def test_set_parent_doesNotMutate(self):
        u = AtomicNode("u")
        c_domain = Domain("c", u)
        c_node = DerivationNode([c_domain], [])
        p_domain = Domain("p", u)
        p_node = DerivationNode([p_domain], [])
        new_c = c_node.set_parent(p_node)
        self.assertExpectedInline(str(c_node.parent), """None""")

    def test_set_parent_setsParentOfEntireSubtree(self):
        u = AtomicNode("u")
        c_domain = Domain("c", u)
        c_node = DerivationNode([c_domain], [])

        gc1_domain = Domain("gc1", u)
        gc1_node = DerivationNode([gc1_domain], [])

        gc2_domain = Domain("gc2", u)
        gc2_node = DerivationNode([gc2_domain], [])

        gc3_domain = Domain("gc3", u)
        gc3_node = DerivationNode([gc3_domain], [])

        c_node.children = OrderedSet([gc1_node, gc2_node, gc3_node])

        p_domain = Domain("p", u)
        p_node = DerivationNode([p_domain], [])
        new_c = c_node.set_parent(p_node)
        self.assertExpectedInline(
            str(new_c),
            """\
[c] // hidden: []
	[gc1] // hidden: []
	[gc2] // hidden: []
	[gc3] // hidden: []""",
        )

    def test_add_child_addsChildToNode(self):
        u = AtomicNode("u")

        p_domain = Domain("p", u)
        p_node = DerivationNode([p_domain], [])

        c_domain = Domain("c", u)
        c_node = DerivationNode([c_domain], [])

        p_new = p_node.add_child(p_node, c_node)
        self.assertExpectedInline(
            str(p_new),
            """\
[p] // hidden: []
	[c] // hidden: []""",
        )

    def test_add_child_doesNotMutate(self):
        u = AtomicNode("u")

        p_domain = Domain("p", u)
        p_node = DerivationNode([p_domain], [])

        c_domain = Domain("c", u)
        c_node = DerivationNode([c_domain], [])

        p_new = c_node.add_child(p_node, c_node)
        self.assertTrue(len(p_node.children) == 0)

    def test_add_child_setsParentOfChild(self):
        u = AtomicNode("u")

        p_domain = Domain("p", u)
        p_node = DerivationNode([p_domain], [])

        c_domain = Domain("c", u)
        c_node = DerivationNode([c_domain], [])

        p_new = p_node.add_child(p_node, c_node)
        self.assertExpectedInline(
            str(p_new.children[0].parent),
            """\
[p] // hidden: []
	[c] // hidden: []""",
        )

    def test_add_child_addsChildToNodeWithExistingChildren(self):
        u = AtomicNode("u")

        p_domain = Domain("p", u)
        p_node = DerivationNode([p_domain], [])

        c1_domain = Domain("c1", u)
        c1_node = DerivationNode([c1_domain], [])

        c2_domain = Domain("c2", u)
        c2_node = DerivationNode([c2_domain], [])

        p_node.children = OrderedSet([c1_node])

        p_new = p_node.add_child(p_node, c2_node)
        self.assertExpectedInline(
            str(p_new),
            """\
[p] // hidden: []
	[c1] // hidden: []
	[c2] // hidden: []""",
        )

    def test_add_child_addsChildToNodeDeeperInTree(self):
        u = AtomicNode("u")

        p_domain = Domain("p", u)
        p_node = DerivationNode([p_domain], [])

        c_domain = Domain("c", u)
        c_node = DerivationNode([c_domain], [])

        gc_domain = Domain("gc", u)
        gc_node = DerivationNode([gc_domain], [])

        p_node.children = OrderedSet([c_node])
        p_new = p_node.add_child(c_node, gc_node)
        self.assertExpectedInline(
            str(p_new),
            """\
[p] // hidden: []
	[c] // hidden: []
		[gc] // hidden: []""",
        )

    def test_add_child_raisesExceptionIfChildAlreadyInTree(self):
        u = AtomicNode("u")

        p_domain = Domain("p", u)
        p_node = DerivationNode([p_domain], [])

        c_domain = Domain("c", u)
        c_node = DerivationNode([c_domain], [])

        p_node.children = OrderedSet([c_node])

        self.assertExpectedRaisesInline(
            NodeIsAlreadyChildOfParentException,
            lambda: p_node.add_child(p_node, c_node),
            """Node c is already a child of p""" "",
        )

    def test_add_children_successfullyAddsChildrenToNode(self):
        u = AtomicNode("u")

        p_domain = Domain("p", u)
        p_node = DerivationNode([p_domain], [])

        c1_domain = Domain("c1", u)
        c1_node = DerivationNode([c1_domain], [])

        c2_domain = Domain("c2", u)
        c2_node = DerivationNode([c2_domain], [])

        p_new = p_node.add_children(p_node, [c1_node, c2_node])
        self.assertExpectedInline(
            str(p_new),
            """\
[p] // hidden: []
	[c1] // hidden: []
	[c2] // hidden: []""",
        )

    def test_add_children_doesNotMutate(self):
        u = AtomicNode("u")

        p_domain = Domain("p", u)
        p_node = DerivationNode([p_domain], [])

        c1_domain = Domain("c1", u)
        c1_node = DerivationNode([c1_domain], [])

        c2_domain = Domain("c2", u)
        c2_node = DerivationNode([c2_domain], [])

        p_new = p_node.add_children(p_node, [c1_node, c2_node])

        self.assertTrue(len(p_node.children) == 0)

    def test_add_children_setsParentOfChildren(self):
        u = AtomicNode("u")

        p_domain = Domain("p", u)
        p_node = DerivationNode([p_domain], [])

        c1_domain = Domain("c1", u)
        c1_node = DerivationNode([c1_domain], [])

        c2_domain = Domain("c2", u)
        c2_node = DerivationNode([c2_domain], [])

        p_new = p_node.add_children(p_node, [c1_node, c2_node])
        self.assertExpectedInline(
            str(p_new.children[0].parent),
            """\
[p] // hidden: []
	[c1] // hidden: []
	[c2] // hidden: []""",
        )

    def test_add_children_addsChildrenToNodeWithExistingChildren(self):
        u = AtomicNode("u")

        p_domain = Domain("p", u)
        p_node = DerivationNode([p_domain], [])

        c1_domain = Domain("c1", u)
        c1_node = DerivationNode([c1_domain], [])

        c2_domain = Domain("c2", u)
        c2_node = DerivationNode([c2_domain], [])

        p_node.children = OrderedSet([c1_node])

        p_new = p_node.add_children(p_node, [c2_node])
        self.assertExpectedInline(
            str(p_new),
            """\
[p] // hidden: []
	[c1] // hidden: []
	[c2] // hidden: []""",
        )

    def test_add_children_addsChildrenToNodeDeeperInTree(self):
        u = AtomicNode("u")

        p_domain = Domain("p", u)
        p_node = DerivationNode([p_domain], [])

        c_domain = Domain("c", u)
        c_node = DerivationNode([c_domain], [])

        gc1_domain = Domain("gc1", u)
        gc1_node = DerivationNode([gc1_domain], [])

        gc2_domain = Domain("gc2", u)
        gc2_node = DerivationNode([gc2_domain], [])

        p_node.children = OrderedSet([c_node])
        p_new = p_node.add_children(c_node, [gc1_node, gc2_node])
        self.assertExpectedInline(
            str(p_new),
            """\
[p] // hidden: []
	[c] // hidden: []
		[gc1] // hidden: []
		[gc2] // hidden: []""",
        )

    def test_add_children_raisesExceptionIfChildAlreadyInTree(self):
        u = AtomicNode("u")

        p_domain = Domain("p", u)
        p_node = DerivationNode([p_domain], [])

        c1_domain = Domain("c1", u)
        c1_node = DerivationNode([c1_domain], [])

        c2_domain = Domain("c2", u)
        c2_node = DerivationNode([c2_domain], [])

        p_node.children = OrderedSet([c1_node])

        self.assertExpectedRaisesInline(
            NodeIsAlreadyChildOfParentException,
            lambda: p_node.add_children(p_node, [c1_node, c2_node]),
            """Node c1 is already a child of p""" "",
        )

    def test_merge_subtree_mergesSubtreeIntoNode(self):
        u = AtomicNode("u")

        p_domain = Domain("p", u)
        p_node = DerivationNode([p_domain], [])

        c1_domain = Domain("c1", u)
        c1_node = DerivationNode([c1_domain], [])

        p_node = p_node.set_children([c1_node])

        p_double_node = DerivationNode([p_domain], [])

        c2_domain = Domain("c2", u)
        c2_node = DerivationNode([c2_domain], [])

        p_double_node = p_double_node.set_children([c2_node])
        p_new = p_node.merge_subtree(p_double_node)

        self.assertExpectedInline(
            str(p_new),
            """\
[p] // hidden: []
	[c1] // hidden: []
	[c2] // hidden: []""",
        )

    def test_merge_subtree_doesNotMutate(self):
        u = AtomicNode("u")

        p_domain = Domain("p", u)
        p_node = DerivationNode([p_domain], [])

        c1_domain = Domain("c1", u)
        c1_node = DerivationNode([c1_domain], [])

        p_node = p_node.set_children([c1_node])

        p_double_node = DerivationNode([p_domain], [])

        c2_domain = Domain("c2", u)
        c2_node = DerivationNode([c2_domain], [])

        p_double_node = p_double_node.set_children([c2_node])
        p_new = p_node.merge_subtree(p_double_node)

        self.assertExpectedInline(
            str(p_node),
            """\
[p] // hidden: []
	[c1] // hidden: []""",
        )

    def test_remove_child_removesChildFromNode(self):
        u = AtomicNode("u")

        p_domain = Domain("p", u)
        p_node = DerivationNode([p_domain], [])

        c_domain = Domain("c", u)
        c_node = DerivationNode([c_domain], [])

        p_node.children = OrderedSet([c_node])

        p_new = p_node.remove_child(p_node, c_node)
        self.assertExpectedInline(str(p_new), """[p] // hidden: []""")

    def test_remove_child_doesNotMutate(self):
        u = AtomicNode("u")

        p_domain = Domain("p", u)
        p_node = DerivationNode([p_domain], [])

        c_domain = Domain("c", u)
        c_node = DerivationNode([c_domain], [])

        p_node.children = OrderedSet([c_node])

        p_new = p_node.remove_child(p_node, c_node)
        self.assertTrue(len(p_node.children) == 1)

    def test_remove_children_removesChildrenFromNode(self):
        u = AtomicNode("u")

        p_domain = Domain("p", u)
        p_node = DerivationNode([p_domain], [])

        c1_domain = Domain("c1", u)
        c1_node = DerivationNode([c1_domain], [])

        c2_domain = Domain("c2", u)
        c2_node = DerivationNode([c2_domain], [])

        p_node.children = OrderedSet([c1_node, c2_node])

        p_new = p_node.remove_children(p_node, [c1_node, c2_node])
        self.assertExpectedInline(str(p_new), """[p] // hidden: []""")

    def test_remove_children_doesNotMutate(self):
        u = AtomicNode("u")

        p_domain = Domain("p", u)
        p_node = DerivationNode([p_domain], [])

        c1_domain = Domain("c1", u)
        c1_node = DerivationNode([c1_domain], [])

        c2_domain = Domain("c2", u)
        c2_node = DerivationNode([c2_domain], [])

        p_node.children = OrderedSet([c1_node, c2_node])

        p_new = p_node.remove_children(p_node, [c1_node, c2_node])
        self.assertTrue(len(p_node.children) == 2)

    def test_find_node_with_domains_returnsNodeIfNodeInTree(self):
        u = AtomicNode("u")
        v = AtomicNode("v")
        w = AtomicNode("w")
        a = Domain("a", u)
        b = Domain("b", v)
        c = Domain("c", w)

        root = DerivationNode.create_root([a, b, c])
        node_a = root.children[1]

        self.assertEqual(root.find_node_with_domains([a]), node_a)

    def test_find_node_with_domains_returnsNoneIfNodeNotInTree(self):
        u = AtomicNode("u")
        v = AtomicNode("v")
        w = AtomicNode("w")
        a = Domain("a", u)
        b = Domain("b", v)
        c = Domain("c", w)

        root = DerivationNode.create_root([a, b, c])
        z = Domain("z", w)

        self.assertEqual(root.find_node_with_domains([z]), None)

    def test_find_root_of_tree_successfullyFindsRootOfTree(self):
        u = AtomicNode("u")
        v = AtomicNode("v")
        w = AtomicNode("w")
        a = Domain("a", u)
        b = Domain("b", v)
        c = Domain("c", w)

        root = DerivationNode.create_root([a, b, c])
        node_a = root.children[1]

        self.assertEqual(node_a.find_root_of_tree(), root)

    def test_to_intermediate_representation_returnsCorrectRepresentationWhenNodeHasNoChildren(
        self,
    ):
        u = AtomicNode("u")
        a = Domain("a", u)
        root = DerivationNode.create_root([a])

        self.assertExpectedInline(
            str(root.children[1].to_intermediate_representation()), """[GET <[a]>]"""
        )

    def test_to_intermediate_representation_returnsCorrectRepresentationWhenNodeHasOneChild(
        self,
    ):
        u = AtomicNode("u")
        v = AtomicNode("v")

        u.id_prefix = 0
        v.id_prefix = 0

        e = SchemaEdge(u, v, Cardinality.MANY_TO_ONE)

        a = Domain("a", u)
        b = Domain("b", v)

        b_node = DerivationNode(
            [b], [StartTraversal([a]), Traverse(e), EndTraversal([b])]
        )
        root = DerivationNode.create_root([a])
        a_node = root.children[1]
        root = root.add_child(a_node, b_node)

        self.assertExpectedInline(
            str(root.children[1].to_intermediate_representation()),
            """[GET <[a]>, CAL, STT <[a]>, TRV <u ---> v, []>, ENT <[b]>, RET]""",
        )

    def test_to_intermediate_representation_returnsCorrectRepresentationWhenNodeHasMultipleChildren(
        self,
    ):
        u = AtomicNode("u")
        v = AtomicNode("v")
        w = AtomicNode("w")

        u.id_prefix = 0
        v.id_prefix = 0
        w.id_prefix = 0

        e_u_to_v = SchemaEdge(u, v, Cardinality.MANY_TO_ONE)
        e_u_to_w = SchemaEdge(u, w, Cardinality.MANY_TO_ONE)

        a = Domain("a", u)
        b = Domain("b", v)
        b_node = DerivationNode(
            [b], [StartTraversal([a]), Traverse(e_u_to_v), EndTraversal([b])]
        )
        c = Domain("c", w)
        c_node = DerivationNode(
            [c], [StartTraversal([a]), Traverse(e_u_to_w), EndTraversal([c])]
        )
        root = DerivationNode.create_root([a])
        a_node = root.children[1]
        root = root.add_children(a_node, [b_node, c_node])
        a_node = root.children[1]

        self.assertExpectedInline(
            str(a_node.to_intermediate_representation()),
            """[GET <[a]>, CAL, STT <[a]>, TRV <u ---> v, []>, ENT <[b]>, RST, STT <[a]>, TRV <u ---> w, []>, ENT <[c]>, MER, RET]""",
        )

    def test_to_intermediate_representation_returnsCorrectRepresentationWhenNodeHasGrandchildren(
        self,
    ):
        u = AtomicNode("u")
        v = AtomicNode("v")
        w = AtomicNode("w")

        u.id_prefix = 0
        v.id_prefix = 0
        w.id_prefix = 0

        e_u_to_v = SchemaEdge(u, v, Cardinality.MANY_TO_ONE)
        e_v_to_w = SchemaEdge(v, w, Cardinality.MANY_TO_ONE)

        a = Domain("a", u)
        b = Domain("b", v)
        c = Domain("c", w)

        b_node = DerivationNode(
            [b], [StartTraversal([a]), Traverse(e_u_to_v), EndTraversal([b])]
        )
        c_node = DerivationNode(
            [c], [StartTraversal([b]), Traverse(e_v_to_w), EndTraversal([c])]
        )

        root = DerivationNode.create_root([a])
        a_node = root.children[1]
        root = root.add_child(a_node, b_node)
        root = root.add_child(b_node, c_node)
        a_node = root.children[1]

        self.assertExpectedInline(
            str(a_node.to_intermediate_representation()),
            """[GET <[a]>, CAL, STT <[a]>, TRV <u ---> v, []>, ENT <[b]>, CAL, STT <[b]>, TRV <v ---> w, []>, ENT <[c]>, RET, RET]""",
        )

    def test_to_intermediate_representation_returnsCorrectRepresentationWhenRootHasMultipleKeys(
        self,
    ):
        u = AtomicNode("u")
        v = AtomicNode("v")
        x = AtomicNode("x")
        y = AtomicNode("y")

        uv = SchemaNode.product([u, v])
        u.id_prefix = 0
        v.id_prefix = 0
        x.id_prefix = 0
        y.id_prefix = 0

        e_uv_to_x = SchemaEdge(uv, x, Cardinality.MANY_TO_ONE)
        e_u_to_y = SchemaEdge(u, y, Cardinality.MANY_TO_ONE)

        a = Domain("a", u)
        b = Domain("b", v)
        c = Domain("c", x)
        d = Domain("d", y)

        c_node = DerivationNode(
            [c], [StartTraversal([a, b]), Traverse(e_uv_to_x), EndTraversal([c])]
        )
        d_node = DerivationNode(
            [d], [StartTraversal([a]), Traverse(e_u_to_y), EndTraversal([d])]
        )

        root = DerivationNode.create_root([a, b])
        root = root.insert_key([a, b])
        a_node = root.children[1]
        ab_node = root.children[3]

        root = root.add_child(ab_node, c_node)
        root = root.add_child(a_node, d_node)

        self.assertExpectedInline(
            str(root.to_intermediate_representation()),
            """[GET <[]>, GET <[a]>, CAL, STT <[a]>, TRV <u ---> y, []>, ENT <[d]>, RET, MER, GET <[b]>, MER, GET <[a, b]>, CAL, STT <[a, b]>, TRV <u;v ---> x, []>, ENT <[c]>, RET, MER]""",
        )

    def test_to_intermediate_representation_returnsCorrectRepresentationWhenTreeHasHiddenKeys(
        self,
    ):
        u = AtomicNode("u")
        v = AtomicNode("v")
        w = AtomicNode("w")

        u.id_prefix = 0
        v.id_prefix = 0
        w.id_prefix = 0

        e_u_to_v = SchemaEdge(u, v, Cardinality.ONE_TO_MANY)

        a = Domain("a", u)
        b = Domain("b", v)
        b1 = Domain("b1", v)

        m = Mapping.create_mapping_from_edge(e_u_to_v, [b1])

        b_node = DerivationNode(
            [b],
            [StartTraversal([a]), Traverse(m), EndTraversal([b])],
            [b1],
        )

        root = DerivationNode.create_root([a])
        a_node = root.children[1]
        root = root.add_child(a_node, b_node)
        a_node = root.children[1]

        self.assertExpectedInline(
            str(a_node.to_intermediate_representation()),
            """[GET <[a]>, CAL, STT <[a]>, TRV <u <--- v, [b1]>, ENT <[b]>, RET]""",
        )

    def test_hideSuccessfullyHidesColumnForDerivationNode(self):
        u = AtomicNode("u")
        v = AtomicNode("v")
        w = AtomicNode("w")

        u.id_prefix = 0
        v.id_prefix = 0
        w.id_prefix = 0

        uv = SchemaNode.product([u, v])

        e_u_to_v = SchemaEdge(u, v, Cardinality.MANY_TO_ONE)
        e_uv_to_w = SchemaEdge(uv, w, Cardinality.MANY_TO_ONE)

        a = Domain("a", u)
        b = Domain("b", v)
        c = Domain("c", w)

        ab_node = DerivationNode(
            [a, b], [StartTraversal([a]), Traverse(e_u_to_v), EndTraversal([b])]
        )

        c_node = DerivationNode(
            [c], [StartTraversal([a, b]), Traverse(e_uv_to_w), EndTraversal([c])]
        )

        root = DerivationNode.create_root([a])
        a_node = root.children[1]
        root = root.add_child(a_node, ab_node).add_child(ab_node, c_node)
        a_node = root.children[1]
        ab_node = a_node.children[0]

        hidden = ab_node.hide(a)
        self.assertExpectedInline(
            str(hidden),
            """\
[b] // hidden: []
	[a, b] // hidden: [a]
		[c] // hidden: []""",
        )

    def test_hide_doesNotMutate(self):
        u = AtomicNode("u")
        v = AtomicNode("v")
        w = AtomicNode("w")

        u.id_prefix = 0
        v.id_prefix = 0
        w.id_prefix = 0

        uv = SchemaNode.product([u, v])

        e_u_to_v = SchemaEdge(u, v, Cardinality.MANY_TO_ONE)
        e_uv_to_w = SchemaEdge(uv, w, Cardinality.MANY_TO_ONE)

        a = Domain("a", u)
        b = Domain("b", v)
        c = Domain("c", w)

        ab_node = DerivationNode(
            [a, b], [StartTraversal([a]), Traverse(e_u_to_v), EndTraversal([b])]
        )

        c_node = DerivationNode(
            [c], [StartTraversal([a, b]), Traverse(e_uv_to_w), EndTraversal([c])]
        )

        root = DerivationNode.create_root([a])
        a_node = root.children[1]
        root = root.add_child(a_node, ab_node).add_child(ab_node, c_node)
        a_node = root.children[1]
        ab_node = a_node.children[0]

        hidden = ab_node.hide(a)
        self.assertExpectedInline(
            str(ab_node),
            """\
[a, b] // hidden: []
	[c] // hidden: []""",
        )

    def test_hide_successfullyTurnsKeysIntoHiddenKeys(self):
        u = AtomicNode("u")
        v = AtomicNode("v")
        w = AtomicNode("w")

        u.id_prefix = 0
        v.id_prefix = 0
        w.id_prefix = 0

        uv = SchemaNode.product([u, v])

        e_u_to_v = SchemaEdge(u, v, Cardinality.MANY_TO_ONE)
        e_uv_to_w = SchemaEdge(uv, w, Cardinality.MANY_TO_ONE)

        a = Domain("a", u)
        b = Domain("b", v)
        c = Domain("c", w)

        ab_node = DerivationNode(
            [a, b], [StartTraversal([a]), Traverse(e_u_to_v), EndTraversal([b])]
        )

        c_node = DerivationNode(
            [c], [StartTraversal([a, b]), Traverse(e_uv_to_w), EndTraversal([c])]
        )

        root = DerivationNode.create_root([a])
        a_node = root.children[1]
        root = root.add_child(a_node, ab_node).add_child(ab_node, c_node)
        a_node = root.children[1]

        hidden = a_node.hide(a)
        self.assertExpectedInline(
            str(hidden),
            """\
[] // hidden: []
	[a] // hidden: [a]
		[b] // hidden: []
			[a, b] // hidden: [a]
				[c] // hidden: []""",
        )
        self.assertTrue(hidden.children[0].is_hidden_key_column())

    def test_hide_successfullyTurnsValueColumnIntoDerivationNode(self):
        u = AtomicNode("u")
        v = AtomicNode("v")
        w = AtomicNode("w")

        u.id_prefix = 0
        v.id_prefix = 0
        w.id_prefix = 0

        e_u_to_v = SchemaEdge(u, v, Cardinality.MANY_TO_ONE)
        e_v_to_w = SchemaEdge(v, w, Cardinality.MANY_TO_ONE)

        a = Domain("a", u)
        b = Domain("b", v)
        c = Domain("c", w)

        root = DerivationNode.create_root([a])
        a_node = root.children[1]
        b_node = ColumnNode(
            b, Val(), [StartTraversal([a]), Traverse(e_u_to_v), EndTraversal([b])]
        )
        c_node = ColumnNode(
            c, Val(), [StartTraversal([b]), Traverse(e_v_to_w), EndTraversal([c])]
        )
        root = root.add_child(a_node, b_node).add_child(b_node, c_node)
        a_node = root.children[1]
        b_node = a_node.children[0]

        hidden = b_node.hide(b)
        self.assertExpectedInline(
            str(hidden),
            """\
[b] // hidden: []
	[c] // hidden: []""",
        )
        self.assertFalse(hidden.is_val_column())

    def test_hide_key_createsHiddenKeyIfOneDoesNotExist(self):
        u = AtomicNode("u")
        v = AtomicNode("v")

        a = Domain("a", u)
        b = Domain("b", v)

        root = DerivationNode.create_root([a, b])
        a_node = root.children[1]
        hidden = root.hide(a_node)

        self.assertExpectedInline(
            str(hidden),
            """\
[b] // hidden: []
	[] // hidden: []
		[a] // hidden: [a]
	[b] // hidden: []""",
        )

    def test_hide_mergesSubtrees(self):
        u = AtomicNode("u")  # a
        v = AtomicNode("v")  # b
        w = AtomicNode("w")  # c
        x = AtomicNode("x")  # d

        uv = SchemaNode.product([u, v])

        u.id_prefix = 0
        v.id_prefix = 0
        w.id_prefix = 0

        e_u_to_w = SchemaEdge(u, w, Cardinality.MANY_TO_ONE)
        e_uv_to_x = SchemaEdge(uv, x, Cardinality.MANY_TO_ONE)

        a = Domain("a", u)
        b = Domain("b", v)
        c = Domain("c", w)
        d = Domain("d", x)

        c_node = ColumnNode(
            c, Val(), [StartTraversal([a]), Traverse(e_u_to_w), EndTraversal([c])]
        )
        d_node = ColumnNode(
            d, Val(), [StartTraversal([a, b]), Traverse(e_uv_to_x), EndTraversal([d])]
        )
        root = DerivationNode.create_root([a, b])
        a_node = root.children[1]
        b_node = root.children[2]
        root = root.insert_key([a, b])
        ab_node = root.children[3]
        root = root.add_child(a_node, c_node).add_child(ab_node, d_node)

        hidden = root.hide(b_node)

        self.assertExpectedInline(
            str(hidden),
            """\
[a] // hidden: []
	[] // hidden: []
		[b] // hidden: [b]
	[a] // hidden: []
		[c] // hidden: []
		[a, b] // hidden: [b]
			[d] // hidden: []""",
        )

    def test_show_key_splitsSubtreesIntoWithAndWithoutKey(self):
        u = AtomicNode("u")
        v = AtomicNode("v")
        w = AtomicNode("w")
        x = AtomicNode("x")

        h = AtomicNode("h")
        hid = Domain("hid", h)

        a = Domain("a", u)
        b = Domain("b", v)
        c = Domain("c", w)
        d = Domain("d", x)

        b_node = DerivationNode([b], [])
        c_node = DerivationNode([c], [])
        d_node = DerivationNode([d], [], [hid])

        root = DerivationNode.create_root([a])
        a_node = root.children[1]
        root = (
            root.add_child(a_node, b_node)
            .add_child(b_node, c_node)
            .add_child(b_node, d_node)
        )
        a_node = root.children[1]
        b_node = a_node.children[0]

        with_hk, without_hk = b_node.show_key(hid)
        self.assertExpectedInline(
            str(with_hk),
            """\
[b, hid] // hidden: []
	[d] // hidden: []""",
        )
        self.assertExpectedInline(
            str(without_hk),
            """\
[b] // hidden: []
	[c] // hidden: []""",
        )

    def test_show_key_doesNotMutate(self):
        u = AtomicNode("u")
        v = AtomicNode("v")
        w = AtomicNode("w")
        x = AtomicNode("x")

        h = AtomicNode("h")
        hid = Domain("hid", h)

        a = Domain("a", u)
        b = Domain("b", v)
        c = Domain("c", w)
        d = Domain("d", x)

        b_node = DerivationNode([b], [])
        c_node = DerivationNode([c], [])
        d_node = DerivationNode([d], [], [hid])

        root = DerivationNode.create_root([a])
        a_node = root.children[1]
        root = (
            root.add_child(a_node, b_node)
            .add_child(b_node, c_node)
            .add_child(b_node, d_node)
        )
        a_node = root.children[1]
        b_node = a_node.children[0]

        _ = b_node.show_key(hid)

        self.assertExpectedInline(
            str(b_node),
            """\
[b] // hidden: []
	[c] // hidden: []
	[d] // hidden: [hid]""",
        )

    def test_show_val_convertsDerivationNodeIntoVal(self):
        u = AtomicNode("u")
        v = AtomicNode("v")
        w = AtomicNode("w")

        u.id_prefix = 0
        v.id_prefix = 0
        w.id_prefix = 0

        a = Domain("a", u)
        b = Domain("b", v)
        c = Domain("c", w)

        root = DerivationNode.create_root([a])
        a_node = root.children[1]
        b_node = DerivationNode([b], [])
        c_node = DerivationNode([c], [])
        root = root.add_child(a_node, b_node).add_child(b_node, c_node)
        a_node = root.children[1]
        b_node = a_node.children[0]

        val = b_node.show_val(c)
        self.assertExpectedInline(
            str(val),
            """\
[b] // hidden: []
	[c] // hidden: []""",
        )
        self.assertTrue(val.children[0].is_val_column())

    def test_equate_internal_doesNothingIfNeitherKeyAppearsInNode(self):
        u = AtomicNode("u")
        v = AtomicNode("v")
        w = AtomicNode("w")

        a1 = Domain("a1", u)
        a2 = Domain("a2", u)

        b = Domain("b", v)
        b_node = DerivationNode([b], [])

        c = Domain("c", w)
        c_node = DerivationNode([c], [])

        root = DerivationNode.create_root([a1, a2])
        a1_node = root.children[1]
        root = root.add_child(a1_node, b_node).add_child(b_node, c_node)
        a1_node = root.children[1]
        b_node = a1_node.children[0]

        equated = b_node.equate_internal(a1, a2, [a1, a2])
        self.assertExpectedInline(
            str(equated),
            """\
[b] // hidden: []
	[c] // hidden: []""",
        )

    def test_equate_internal_performsSubstitutionIfOnlySecondKeyAppearsInNode(self):
        u = AtomicNode("u")
        v = AtomicNode("v")
        w = AtomicNode("w")

        a1 = Domain("a1", u)
        a2 = Domain("a2", u)

        b = Domain("b", v)
        a2b_node = DerivationNode([a2, b], [])

        c = Domain("c", w)
        c_node = DerivationNode([c], [])

        root = DerivationNode.create_root([a1, a2])
        a2_node = root.children[2]
        root = root.add_child(a2_node, a2b_node).add_child(a2b_node, c_node)
        a2_node = root.children[2]
        a2b_node = a2_node.children[0]

        equated = a2b_node.equate_internal(a1, a2, [a1, a2])
        self.assertExpectedInline(
            str(equated),
            """\
[a1, b] // hidden: []
	[c] // hidden: []""",
        )

    def test_equate_internal_eliminatesSecondKeyIfBothFirstAndSecondKeyAppearInNode(
        self,
    ):
        u = AtomicNode("u")
        v = AtomicNode("v")
        w = AtomicNode("w")

        a1 = Domain("a1", u)
        a2 = Domain("a2", u)

        b = Domain("b", v)
        a1a2b_node = DerivationNode([a1, a2, b], [])

        c = Domain("c", w)
        c_node = DerivationNode([c], [])

        root = DerivationNode.create_root([a1, a2])
        root = root.insert_key([a1, a2])

        a1a2_node = root.children[3]
        root = root.add_child(a1a2_node, a1a2b_node).add_child(a1a2b_node, c_node)
        a1a2_node = root.children[3]
        a1a2b_node = a1a2_node.children[0]

        equated = a1a2b_node.equate_internal(a1, a2, [a1, a2])
        self.assertExpectedInline(
            str(equated),
            """\
[a1, b] // hidden: []
	[c] // hidden: []""",
        )

    def test_equate_doesNotMutate(self):
        u = AtomicNode("u")
        v = AtomicNode("v")
        w = AtomicNode("w")

        a1 = Domain("a1", u)
        a2 = Domain("a2", u)

        b = Domain("b", v)
        a1a2b_node = DerivationNode([a1, a2, b], [])

        c = Domain("c", w)
        c_node = DerivationNode([c], [])

        root = DerivationNode.create_root([a1, a2])
        root = root.insert_key([a1, a2])

        a1a2_node = root.children[3]
        root = root.add_child(a1a2_node, a1a2b_node).add_child(a1a2b_node, c_node)
        a1a2_node = root.children[3]
        a1a2b_node = a1a2_node.children[0]

        equated = a1a2b_node.equate_internal(a1, a2, [a1, a2])
        self.assertExpectedInline(
            str(a1a2b_node),
            """\
[a1, a2, b] // hidden: []
	[c] // hidden: []""",
        )

    def test_equate_modifiesIntermediateRepresentation(self):
        u = AtomicNode("u")
        v = AtomicNode("v")
        w = AtomicNode("w")

        u.id_prefix = 0
        v.id_prefix = 0
        w.id_prefix = 0

        uu = SchemaNode.product([u, u])
        uuv = SchemaNode.product([u, u, v])

        a1 = Domain("a1", u)
        a2 = Domain("a2", u)

        b = Domain("b", v)

        edge = SchemaEdge(uu, v, Cardinality.MANY_TO_ONE)
        edge2 = SchemaEdge(uuv, w, Cardinality.MANY_TO_ONE)

        a1a2b_node = DerivationNode(
            [a1, a2, b], [StartTraversal([a1, a2]), Traverse(edge), EndTraversal([b])]
        )

        c = Domain("c", w)
        c_node = DerivationNode(
            [c], [StartTraversal([a1, a2, b]), Traverse(edge2), EndTraversal([c])]
        )

        root = DerivationNode.create_root([a1, a2])
        root = root.insert_key([a1, a2])

        a1a2_node = root.children[3]
        root = root.add_child(a1a2_node, a1a2b_node).add_child(a1a2b_node, c_node)
        a1a2_node = root.children[3]
        a1a2b_node = a1a2_node.children[0]

        equated = a1a2b_node.equate_internal(a1, a2, [a1, a2])
        child = equated.children[0]
        self.assertExpectedInline(
            str(child.to_intermediate_representation()),
            """[STT <[a1, b]>, PRJ <u;v, u, [0]>, ENT <[a2]>, STT <[a1, a2, b]>, TRV <u;u;v ---> w, []>, ENT <[c]>, DRP <[a2]>]""",
        )

    def test_rename_doesNothingIfDomainNotInNode(self):
        u = AtomicNode("u")
        v = AtomicNode("v")

        a = Domain("a", u)
        b = Domain("b", v)

        root = DerivationNode.create_root([a])
        a_node = root.children[1]
        b_node = DerivationNode([b], [])
        root = root.add_child(a_node, b_node)
        a_node = root.children[1]
        b_node = a_node.children[0]

        renamed = b_node.rename("c", "d")
        self.assertExpectedInline(str(renamed), """[b] // hidden: []""")

    def test_rename_successfullyRenames(self):
        u = AtomicNode("u")
        v = AtomicNode("v")

        a = Domain("a", u)
        b = Domain("b", v)

        root = DerivationNode.create_root([a])
        a_node = root.children[1]
        b_node = DerivationNode([b], [])
        root = root.add_child(a_node, b_node)
        a_node = root.children[1]
        b_node = a_node.children[0]

        renamed = b_node.rename("b", "c")
        self.assertExpectedInline(str(renamed), """[c] // hidden: []""")

    def test_rename_successfullyRenamesUnderRepresentation(self):
        u = AtomicNode("u")
        v = AtomicNode("v")
        u.id_prefix = 0
        v.id_prefix = 0

        a = Domain("a", u)
        b = Domain("b", v)

        root = DerivationNode.create_root([a])
        a_node = root.children[1]
        edge = SchemaEdge(u, v, Cardinality.MANY_TO_ONE)
        m = Mapping.create_mapping_from_edge(edge)
        b_node = DerivationNode(
            [b], [StartTraversal([a]), Traverse(m), EndTraversal([b])]
        )
        root = root.add_child(a_node, b_node)
        a_node = root.children[1]
        b_node = a_node.children[0]

        renamed = b_node.rename("b", "c")
        self.assertExpectedInline(
            str(renamed.to_intermediate_representation()),
            """[STT <[a]>, TRV <u ---> v, []>, ENT <[c]>]""",
        )

    def test_rename_doesNotMutate(self):
        u = AtomicNode("u")
        v = AtomicNode("v")

        a = Domain("a", u)
        b = Domain("b", v)

        root = DerivationNode.create_root([a])
        a_node = root.children[1]
        b_node = DerivationNode([b], [])
        root = root.add_child(a_node, b_node)
        a_node = root.children[1]
        b_node = a_node.children[0]

        _ = b_node.rename("b", "c")
        self.assertExpectedInline(str(b_node), """[b] // hidden: []""")

    def test_is_value_or_set_of_values_returnsTrueIfNodeIsValue(self):
        u = AtomicNode("u")
        a = Domain("a", u)

        val = ColumnNode(a, Val(), [])

        self.assertTrue(val.is_value_or_set_of_values())

    def test_is_value_or_set_of_values_returnsTrueIfNodeIsSetOfValues(self):
        u = AtomicNode("u")

        a = Domain("a", u)
        b = Domain("b", u)
        c = Domain("c", u)

        a_node = ColumnNode(a, Val(), [])
        b_node = ColumnNode(b, Val(), [])
        c_node = ColumnNode(c, Val(), [])

        set_of_vals = DerivationNode([a, b, c], [])
        set_of_vals = set_of_vals.set_children([a_node, b_node, c_node])

        self.assertTrue(set_of_vals.is_value_or_set_of_values())

    def test_is_value_or_set_of_values_returnsFalseIfNodeIsKey(self):
        u = AtomicNode("u")
        a = Domain("a", u)

        key = ColumnNode(a, Key(), [])

        self.assertFalse(key.is_value_or_set_of_values())

    def test_is_value_or_set_of_values_returnsFalseIfNodeIsHiddenKey(self):
        u = AtomicNode("u")
        a = Domain("a", u)

        key = ColumnNode(a, HiddenKey(), [])

        self.assertFalse(key.is_value_or_set_of_values())

    def test_check_if_value_or_set_of_values_returnsFalseIfNodeIsNotSetOfNodes(self):
        u = AtomicNode("u")
        a = Domain("a", u)
        b = Domain("b", u)
        c = Domain("c", u)

        a_node = ColumnNode(a, Val(), [])
        b_node = DerivationNode([b], [])
        c_node = ColumnNode(c, Val(), [])

        node = DerivationNode([a, b, c], [])
        node = node.set_children([a_node, b_node, c_node])

        self.assertFalse(node.is_value_or_set_of_values())

    def test_path_to_value_returnsPathToValueNode(self):
        u = AtomicNode("u")
        v = AtomicNode("v")

        a = Domain("a", u)
        b = Domain("b", v)

        edge = SchemaEdge(u, v, Cardinality.MANY_TO_ONE)
        b_node = ColumnNode(
            b, Val(), [StartTraversal([a]), Traverse(edge), EndTraversal([b])]
        )

        root = DerivationNode.create_root([a])
        a_node = root.children[1]
        root = root.add_child(a_node, b_node)
        a_node = root.children[1]

        path = a_node.path_to_value(b_node)
        self.assertEqual(path, [a_node, b_node])

    def test_path_to_value_returnsPathToSetOfValues(self):
        u = AtomicNode("u")
        v = AtomicNode("v")
        w = AtomicNode("w")

        vw = SchemaNode.product([v, w])

        a = Domain("a", u)
        b = Domain("b", v)
        c = Domain("c", w)

        edge = SchemaEdge(u, vw, Cardinality.MANY_TO_ONE)

        bc_node = DerivationNode(
            [b, c], [StartTraversal([a]), Traverse(edge), EndTraversal([b, c])]
        )

        b_node = ColumnNode(
            b, Val(), [StartTraversal([b, c]), Project(vw, v, [0]), EndTraversal([b])]
        )
        c_node = ColumnNode(
            c, Val(), [StartTraversal([b, c]), Project(vw, w, [1]), EndTraversal([c])]
        )

        root = DerivationNode.create_root([a])
        a_node = root.children[1]
        root = root.add_child(a_node, bc_node)
        a_node = root.children[1]
        bc_node = a_node.children[0]

        root = root.add_children(bc_node, [b_node, c_node])

        path = root.path_to_value(bc_node)
        self.assertEqual(path, [root, a_node, bc_node])

    def test_intermediate_representation_for_path_returnsEmptyListIfPathEmpty(self):
        self.assertEqual(intermediate_representation_for_path([]), [])

    def test_intermediate_representation_for_path_returnsCorrectIntermediateRepresentationForPath(
        self,
    ):
        u = AtomicNode("u")
        v = AtomicNode("v")

        u.id_prefix = 0
        v.id_prefix = 0

        a = Domain("a", u)
        b = Domain("b", v)

        edge = SchemaEdge(u, v, Cardinality.MANY_TO_ONE)
        b_node = ColumnNode(
            b, Val(), [StartTraversal([a]), Traverse(edge), EndTraversal([b])]
        )

        root = DerivationNode.create_root([a])
        a_node = root.children[1]
        root = root.add_child(a_node, b_node)
        a_node = root.children[1]

        path = a_node.path_to_value(b_node)
        self.assertExpectedInline(
            str(intermediate_representation_for_path(path)),
            """[GET <[a]>, STT <[a]>, TRV <u ---> v, []>, ENT <[b]>]""",
        )

    def test_invert_derivation_path_successfullyInvertsDerivationPath(self):
        u = AtomicNode("u")
        v = AtomicNode("v")
        w = AtomicNode("w")

        u.id_prefix = 0
        v.id_prefix = 0
        w.id_prefix = 0

        a = Domain("a", u)
        b = Domain("b", v)
        c = Domain("c", w)

        edge = SchemaEdge(u, v, Cardinality.MANY_TO_ONE)
        m1 = Mapping.create_mapping_from_edge(edge)
        edge2 = SchemaEdge(v, w, Cardinality.MANY_TO_ONE)
        m2 = Mapping.create_mapping_from_edge(edge2)

        a_node = DerivationNode([a], [Get([a])])
        b_node = DerivationNode(
            [b], [StartTraversal([a]), Traverse(m1), EndTraversal([b])]
        )
        c_node = DerivationNode(
            [c], [StartTraversal([b]), Traverse(m2), EndTraversal([c])]
        )

        a_node = a_node.add_child(a_node, b_node).add_child(b_node, c_node)
        b_node = a_node.children[0]
        c_node = b_node.children[0]
        inverted = invert_derivation_path([a_node, b_node, c_node], frozenset())
        self.assertExpectedInline(
            str(inverted),
            """\
[c] // hidden: []
	[b] // hidden: [v]
		[a] // hidden: [u]""",
        )

    def test_invert_derivation_path_invertsRepresentationSteps(self):
        u = AtomicNode("u")
        v = AtomicNode("v")
        w = AtomicNode("w")

        u.id_prefix = 0
        v.id_prefix = 0
        w.id_prefix = 0

        a = Domain("a", u)
        b = Domain("b", v)
        c = Domain("c", w)

        edge = SchemaEdge(u, v, Cardinality.MANY_TO_ONE)
        m1 = Mapping.create_mapping_from_edge(edge)
        edge2 = SchemaEdge(v, w, Cardinality.MANY_TO_ONE)
        m2 = Mapping.create_mapping_from_edge(edge2)

        a_node = DerivationNode([a], [Get([a])])
        b_node = DerivationNode(
            [b], [StartTraversal([a]), Traverse(m1), EndTraversal([b])]
        )
        c_node = DerivationNode(
            [c], [StartTraversal([b]), Traverse(m2), EndTraversal([c])]
        )

        a_node = a_node.add_child(a_node, b_node).add_child(b_node, c_node)
        b_node = a_node.children[0]
        c_node = b_node.children[0]
        inverted = invert_derivation_path([a_node, b_node, c_node], frozenset())
        self.assertExpectedInline(
            str(inverted.to_intermediate_representation()),
            """[GET <[c]>, CAL, STT <[c]>, TRV <w <--- v, [v]>, ENT <[b]>, CAL, STT <[b]>, TRV <v <--- u, [u]>, ENT <[a]>, RET, RET]"""
            "",
        )

    def test_invert_derivation_path_renamesHiddenKeysToAvoidNamespaceClashes(self):
        u = AtomicNode("u")
        v = AtomicNode("v")
        w = AtomicNode("w")

        u.id_prefix = 0
        v.id_prefix = 0
        w.id_prefix = 0

        a = Domain("a", u)
        b = Domain("b", v)
        c = Domain("c", w)

        edge = SchemaEdge(u, v, Cardinality.MANY_TO_ONE)
        m1 = Mapping.create_mapping_from_edge(edge)
        edge2 = SchemaEdge(v, w, Cardinality.MANY_TO_ONE)
        m2 = Mapping.create_mapping_from_edge(edge2)

        a_node = DerivationNode([a], [Get([a])])
        b_node = DerivationNode(
            [b], [StartTraversal([a]), Traverse(m1), EndTraversal([b])]
        )
        c_node = DerivationNode(
            [c], [StartTraversal([b]), Traverse(m2), EndTraversal([c])]
        )

        a_node = a_node.add_child(a_node, b_node).add_child(b_node, c_node)
        b_node = a_node.children[0]
        c_node = b_node.children[0]
        inverted = invert_derivation_path([a_node, b_node, c_node], frozenset(["u"]))
        self.assertExpectedInline(
            str(inverted),
            """\
[c] // hidden: []
	[b] // hidden: [v]
		[a] // hidden: [u_1]""",
        )

    def test_invert_derivation_path_doesNotInvertChildrenNotOnPath(self):
        u = AtomicNode("u")
        v = AtomicNode("v")
        w = AtomicNode("w")

        u.id_prefix = 0
        v.id_prefix = 0
        w.id_prefix = 0

        a = Domain("a", u)
        b = Domain("b", v)
        c = Domain("c", w)
        d = Domain("d", u)
        e = Domain("e", u)

        edge = SchemaEdge(u, v, Cardinality.MANY_TO_ONE)
        m1 = Mapping.create_mapping_from_edge(edge)
        edge2 = SchemaEdge(v, w, Cardinality.MANY_TO_ONE)
        m2 = Mapping.create_mapping_from_edge(edge2)

        a_node = DerivationNode([a], [Get([a])])
        b_node = DerivationNode(
            [b], [StartTraversal([a]), Traverse(m1), EndTraversal([b])]
        )
        c_node = DerivationNode(
            [c], [StartTraversal([b]), Traverse(m2), EndTraversal([c])]
        )

        d_node = DerivationNode([d], [])
        e_node = DerivationNode([e], [])

        a_node = (
            a_node.add_child(a_node, b_node)
            .add_child(b_node, c_node)
            .add_child(b_node, d_node)
            .add_child(c_node, e_node)
        )
        b_node = a_node.children[0]
        c_node = b_node.children[0]
        inverted = invert_derivation_path([a_node, b_node, c_node], frozenset())

        self.assertExpectedInline(
            str(inverted),
            """\
[c] // hidden: []
	[e] // hidden: []
	[b] // hidden: [v]
		[d] // hidden: []
		[a] // hidden: [u]""",
        )

    def test_invert_derivation_path_doesNotMutate(self):
        u = AtomicNode("u")
        v = AtomicNode("v")
        w = AtomicNode("w")

        u.id_prefix = 0
        v.id_prefix = 0
        w.id_prefix = 0

        a = Domain("a", u)
        b = Domain("b", v)
        c = Domain("c", w)

        edge = SchemaEdge(u, v, Cardinality.MANY_TO_ONE)
        m1 = Mapping.create_mapping_from_edge(edge)
        edge2 = SchemaEdge(v, w, Cardinality.MANY_TO_ONE)
        m2 = Mapping.create_mapping_from_edge(edge2)

        a_node = DerivationNode([a], [Get([a])])
        b_node = DerivationNode(
            [b], [StartTraversal([a]), Traverse(m1), EndTraversal([b])]
        )
        c_node = DerivationNode(
            [c], [StartTraversal([b]), Traverse(m2), EndTraversal([c])]
        )

        a_node = a_node.add_child(a_node, b_node).add_child(b_node, c_node)
        b_node = a_node.children[0]
        c_node = b_node.children[0]
        _ = invert_derivation_path([a_node, b_node, c_node], frozenset())
        self.assertExpectedInline(
            str(a_node),
            """\
[a] // hidden: []
	[b] // hidden: []
		[c] // hidden: []""",
        )

    def test_set_hidden_keys_along_path_avoidsNamespaceClashes(self):
        u = AtomicNode("u")
        v = AtomicNode("v")
        w = AtomicNode("w")

        u.id_prefix = 0
        v.id_prefix = 0
        w.id_prefix = 0

        a = Domain("a", u)
        b = Domain("b", v)
        c = Domain("c", w)

        edge = SchemaEdge(u, v, Cardinality.ONE_TO_MANY)
        m1 = Mapping.create_mapping_from_edge(edge, [Domain("v", v)])
        edge2 = SchemaEdge(v, w, Cardinality.ONE_TO_MANY)
        m2 = Mapping.create_mapping_from_edge(edge2, [Domain("w", v)])

        a_node = DerivationNode([a], [Get([a])])
        b_node = DerivationNode(
            [b], [StartTraversal([a]), Traverse(m1), EndTraversal([b])]
        )
        c_node = DerivationNode(
            [c], [StartTraversal([b]), Traverse(m2), EndTraversal([c])]
        )

        a_node = a_node.add_child(a_node, b_node).add_child(b_node, c_node)
        b_node = a_node.children[0]
        c_node = b_node.children[0]
        hidden_keys_set = set_and_name_hidden_keys_along_path(
            [a_node, b_node, c_node], a_node, frozenset(["v"])
        )
        self.assertExpectedInline(
            str(hidden_keys_set),
            """\
[a] // hidden: []
	[b] // hidden: [v_1]
		[c] // hidden: [w]""",
        )

    def test_set_hidden_keys_along_path_doesNotMutate(self):
        u = AtomicNode("u")
        v = AtomicNode("v")
        w = AtomicNode("w")

        u.id_prefix = 0
        v.id_prefix = 0
        w.id_prefix = 0

        a = Domain("a", u)
        b = Domain("b", v)
        c = Domain("c", w)

        edge = SchemaEdge(u, v, Cardinality.MANY_TO_ONE)
        m1 = Mapping.create_mapping_from_edge(edge)
        edge2 = SchemaEdge(v, w, Cardinality.MANY_TO_ONE)
        m2 = Mapping.create_mapping_from_edge(edge2)

        a_node = DerivationNode([a], [Get([a])])
        b_node = DerivationNode(
            [b], [StartTraversal([a]), Traverse(m1), EndTraversal([b])]
        )
        c_node = DerivationNode(
            [c], [StartTraversal([b]), Traverse(m2), EndTraversal([c])]
        )

        a_node = a_node.add_child(a_node, b_node).add_child(b_node, c_node)
        b_node = a_node.children[0]
        c_node = b_node.children[0]
        _ = set_and_name_hidden_keys_along_path(
            [a_node, b_node, c_node], a_node, frozenset(["v"])
        )
        self.assertExpectedInline(
            str(a_node),
            """\
[a] // hidden: []
	[b] // hidden: []
		[c] // hidden: []""",
        )

    def test_find_column_with_name_successfullyFindsColumnIfExists(self):
        u = AtomicNode("u")
        v = AtomicNode("v")
        w = AtomicNode("w")

        u.id_prefix = 0
        v.id_prefix = 0
        w.id_prefix = 0

        a = Domain("a", u)
        b = Domain("b", v)
        c = Domain("c", w)

        edge = SchemaEdge(u, v, Cardinality.MANY_TO_ONE)
        edge2 = SchemaEdge(v, w, Cardinality.MANY_TO_ONE)

        a_node = DerivationNode([a], [Get([a])])
        b_node = ColumnNode(
            b, Val(), [StartTraversal([a]), Traverse(edge), EndTraversal([b])]
        )
        c_node = DerivationNode(
            [c], [StartTraversal([b]), Traverse(edge2), EndTraversal([c])]
        )

        a_node = a_node.add_child(a_node, b_node).add_child(b_node, c_node)

        self.assertEqual(a_node.find_column_with_name("b"), b_node)

    def test_find_column_with_name_returnsNoneIfColumnDoesNotExist(self):
        u = AtomicNode("u")
        v = AtomicNode("v")

        a = Domain("a", u)
        b = Domain("b", v)

        a_node = DerivationNode([a], [Get([a])])

        self.assertIsNone(a_node.find_column_with_name("b"))

    def test_find_all_keys_in_tree_successfullyFindsAllKeys(self):
        u = AtomicNode("u")
        v = AtomicNode("v")

        a = Domain("a", u)
        b = Domain("b", v)

        root = DerivationNode.create_root([a, b])

        a_node, b_node = root.children[1:]

        keys = root.find_all_keys_in_tree()
        subkeys = a_node.find_all_keys_in_tree()

        self.assertEqual(keys, [a_node, b_node])
        self.assertEqual(subkeys, [a_node])

    def test_find_all_values_in_tree_successfullyFindsAllValues(self):
        u = AtomicNode("u")
        v = AtomicNode("v")
        w = AtomicNode("w")
        x = AtomicNode("x")

        a = Domain("a", u)
        b = Domain("b", v)
        c = Domain("c", w)
        d = Domain("d", x)

        c_node = ColumnNode(c, Val(), [])
        d_node = ColumnNode(d, Val(), [])

        root = DerivationNode.create_root([a, b])

        a_node, b_node = root.children[1:]

        root = root.add_child(a_node, c_node).add_child(b_node, d_node)
        a_node, b_node = root.children[1:]

        values = root.find_all_values_in_tree()
        subvalues = a_node.find_all_values_in_tree()

        self.assertEqual(values, [c_node, d_node])
        self.assertEqual(subvalues, [c_node])

    def test_find_all_hidden_keys_in_tree_successfullyFindsAllHiddenKeys(self):
        u = AtomicNode("u")
        v = AtomicNode("v")

        a = Domain("a", u)
        b = Domain("b", v)
        b_node = ColumnNode(b, HiddenKey(), [])

        root = DerivationNode.create_root([a])

        root = root.add_hidden_key(b)

        hidden = root.find_all_hidden_keys_in_tree()

        self.assertEqual(hidden, [b_node])

    def test_find_splice_point_successfullyFindsSplicePoint(self):
        u = AtomicNode("u")
        v = AtomicNode("v")
        w = AtomicNode("w")
        x = AtomicNode("x")

        a = Domain("a", u)
        b = Domain("b", v)
        c = Domain("c", w)
        d = Domain("d", x)

        a_node = DerivationNode([a], [])
        b_node = DerivationNode([b], [])
        c_node = DerivationNode([c], [])
        d_node = DerivationNode([d], [])

        a_node = a_node.add_child(a_node, b_node).add_child(b_node, c_node)
        path_1 = [a_node, d_node]
        path_2 = [a_node, b_node, d_node]

        splice_point_1 = find_splice_point(a_node, path_1)

        splice_point_2 = find_splice_point(a_node, path_2)

        self.assertEqual(splice_point_1, 0)
        self.assertEqual(splice_point_2, 1)
