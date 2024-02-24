import expecttest

from frontend.domain import Domain
from frontend.mapping import Mapping
from representation.helpers.show_key_in_representation import *
from representation.representation import StartTraversal, Traverse, Expand, EndTraversal
from schema.cardinality import Cardinality
from schema.edge import SchemaEdge
from schema.node import SchemaNode, AtomicNode


class TestShowKeyInRepresentation(expecttest.TestCase):

    def test_find_index_of_instruction_causing_hidden_key_succeedsWhenCausedByTraverse(self):
        u = AtomicNode("u")
        v = AtomicNode("v")
        a = Domain("a", u)
        b = Domain("b", v)
        c = Domain("c", v)
        e = SchemaEdge(u, v, Cardinality.ONE_TO_MANY)
        mapping = Mapping.create_mapping_from_edge(e, [c])
        representation = [StartTraversal([a]), Traverse(mapping), EndTraversal([b])]

        self.assertEqual(find_index_of_instruction_causing_hidden_key(c, representation), 1)

    def test_find_index_of_instruction_causing_hidden_key_succeedsWhenCausedByExpand(self):
        u = AtomicNode("u")
        v = AtomicNode("v")
        uv = SchemaNode.product([u, v])
        a = Domain("a", u)
        b = Domain("b", v)
        c = Domain("c", v)
        representation = [StartTraversal([a]), Expand(u, uv, [0], [c]), EndTraversal([b])]

        self.assertEqual(find_index_of_instruction_causing_hidden_key(c, representation), 1)

    def test_find_index_of_instruction_causing_hidden_key_findsFirstSuchInstruction(self):
        u = AtomicNode("u")
        v = AtomicNode("v")
        uv = SchemaNode.product([u, v])
        a = Domain("a", u)
        b = Domain("b", v)
        c = Domain("c", v)
        e = SchemaEdge(u, v, Cardinality.ONE_TO_MANY)
        mapping = Mapping.create_mapping_from_edge(e, [c])
        representation = [StartTraversal([a]), Traverse(mapping), EndTraversal([b]), StartTraversal([a]), Expand(u, uv, [0], [c]), EndTraversal([b])]
        self.assertEqual(find_index_of_instruction_causing_hidden_key(c, representation), 1)

    def test_find_index_of_instruction_causing_hidden_key_returnsNoneIfNoSuchInstruction(self):
        u = AtomicNode("u")
        v = AtomicNode("v")
        a = Domain("a", u)
        b = Domain("b", v)
        c = Domain("c", v)
        e = SchemaEdge(u, v, Cardinality.MANY_TO_ONE)
        mapping = Mapping.create_mapping_from_edge(e)
        representation = [StartTraversal([a]), Traverse(mapping), EndTraversal([b])]
        self.assertIsNone(find_index_of_instruction_causing_hidden_key(c, representation))

    def test_show_key_in_representation_segment_successfullyShowsKeyWhenOffendingIsTraverse(self):
        u = AtomicNode("u")
        v = AtomicNode("v")
        w = AtomicNode("w")
        u.id_prefix = 0
        v.id_prefix = 0
        w.id_prefix = 0

        a = Domain("a", u)
        c = Domain("c", w)
        c_1 = Domain("c_1", w)

        e1 = SchemaEdge(u, v, Cardinality.MANY_TO_ONE)
        e2 = SchemaEdge(v, w, Cardinality.ONE_TO_MANY)

        m1 = Mapping.create_mapping_from_edge(e1)
        m2 = Mapping.create_mapping_from_edge(e2, [c_1])
        
        representation = [StartTraversal([a]), Traverse(m1), Traverse(m2), EndTraversal([c])]

        result, idx = show_key_in_representation_segment(c_1, representation)
        self.assertExpectedInline(str(result), """[STT <[a, c_1]>, TRV <u;w ---> v;w, []>, TRV <v;w ---> w, []>, ENT <[c]>]""")
        self.assertEqual(idx, 3)

    def test_show_key_in_representation_segment_successfullyShowsKeyWhenOffendingIsExpandAndExpandHasOnlyOneHiddenKey(self):
        u = AtomicNode("u")
        v = AtomicNode("v")
        w = AtomicNode("w")
        vw = SchemaNode.product([v, w])

        u.id_prefix = 0
        v.id_prefix = 0
        w.id_prefix = 0

        a = Domain("a", u)
        b = Domain("b", v)
        c = Domain("c", w)
        c_1 = Domain("c_1", w)

        e1 = SchemaEdge(u, v, Cardinality.MANY_TO_ONE)

        m1 = Mapping.create_mapping_from_edge(e1)

        representation = [StartTraversal([a]), Traverse(m1), Expand(v, vw, [0], [c_1]), EndTraversal([b, c])]

        result, idx = show_key_in_representation_segment(c_1, representation)
        self.assertExpectedInline(str(result), """[STT <[a, c_1]>, TRV <u;w ---> v;w, []>, ENT <[b, c]>]""")


