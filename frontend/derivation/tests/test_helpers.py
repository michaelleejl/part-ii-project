import expecttest

from frontend.derivation.derivation_node import (
    intermediate_representation_for_path,
    DerivationNode,
    ColumnNode,
    invert_derivation_path,
    set_and_name_hidden_keys_along_path,
    find_splice_point,
    compress_path_representation,
)
from representation.mapping import Mapping
from frontend.tables.column_type import Val
from representation.representation import *
from schema.cardinality import Cardinality
from schema.edge import SchemaEdge
from schema.node import AtomicNode, SchemaNode


class TestHelpers(expecttest.TestCase):
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

    def test_compress_path_representation_successfullyCompressesPathRepresentation(
        self,
    ):
        u = AtomicNode("u")
        v = AtomicNode("v")
        w = AtomicNode("w")
        x = AtomicNode("x")

        vw = SchemaNode.product([v, w])

        u.id_prefix = 0
        v.id_prefix = 0
        w.id_prefix = 0
        x.id_prefix = 0

        a = Domain("a", u)
        b = Domain("b", v)
        c = Domain("c", w)
        d = Domain("d", x)

        e1 = SchemaEdge(u, v, Cardinality.MANY_TO_ONE)
        m1 = Mapping.create_mapping_from_edge(e1)
        e2 = SchemaEdge(v, w, Cardinality.MANY_TO_ONE)
        m2 = Mapping.create_mapping_from_edge(e2)
        e3 = SchemaEdge(vw, x, Cardinality.MANY_TO_ONE)
        m3 = Mapping.create_mapping_from_edge(e3)

        root = DerivationNode.create_root([a])
        a_node = root.children[1]
        bc_node = DerivationNode(
            [b, c],
            [
                StartTraversal([a]),
                Traverse(m1),
                EndTraversal([b]),
                StartTraversal([a]),
                Traverse(m2),
                EndTraversal([c]),
                Merge(),
            ],
        )
        d_node = DerivationNode(
            [d], [StartTraversal([b, c]), Traverse(m3), EndTraversal([d])]
        )

        a_node = a_node.add_child(a_node, bc_node).add_child(bc_node, d_node)
        bc_node = a_node.children[0]
        d_node = bc_node.children[0]
        path = [bc_node, d_node]

        print(bc_node.parent)

        compressed = compress_path_representation(path)

        self.assertExpectedInline(
            str(compressed),
            """[STT <[a]>, TRV <u ---> v, []>, ENT <[b]>, STT <[a]>, TRV <v ---> w, []>, ENT <[c]>, MER, STT <[b, c]>, TRV <v;w ---> x, []>, ENT <[d]>, DRP <[b, c]>]""",
        )
