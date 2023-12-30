import expecttest

from schema import SchemaNode, AtomicNode
from union_find.union_find import UnionFind


class TestSchemaNode(expecttest.TestCase):

    def test_schemaNodeEquality_returnsTrue_ifNodesAtomicAndNameAndIdEqual(self):
        u = AtomicNode("name")
        self.assertEqual(u, u)

    def test_schemaNodeEquality_returnsTrue_ifNodesAtomicAndNameEqualButIdNotEqual(self):
        u = AtomicNode("name")
        v = AtomicNode("name")
        self.assertNotEqual(u, v)

    def test_schemaNodeEquality_returnsFalse_ifNodesAtomicAndNameNotEqual(self):
        u = AtomicNode("name")
        v = AtomicNode("name'")
        self.assertNotEqual(u, v)

    def test_schemaNodeEquality_returnsFalse_whenComparingAtomicNodeWithProductNode(self):
        u = AtomicNode("1")
        v = AtomicNode("2")
        w = SchemaNode.product([u, v])
        self.assertNotEqual(u, w)

    def test_schemaNodeEquality_forProductNodes_respectsOrder(self):
        u = AtomicNode("1")
        v = AtomicNode("2")
        w1 = SchemaNode.product([u, v])
        w2 = SchemaNode.product([v, u])
        self.assertNotEqual(w1, w2)

    def test_schemaNodeEquality_returnsFalse_whenComparingUnequalProductNodes(self):
        u = AtomicNode("1")
        v = AtomicNode("2")
        w = AtomicNode("3")
        w1 = SchemaNode.product([u, v])
        w2 = SchemaNode.product([v, w])
        self.assertNotEqual(w1, w2)

    def test_schemaNodeProduct_returnsOriginalNode_whenGivenOneNode(self):
        u = AtomicNode("1")
        w = SchemaNode.product([u])
        self.assertEqual(u, w)

    def test_schemaNodeLTEQ_betweenAtomicNodes_returnsTrueIfNodesEqual(self):
        u = AtomicNode("1")
        self.assertTrue(u <= u)

    def test_schemaNodeLTEQ_betweenAtomicNodes_returnsFalseIfNodesNotEqual(self):
        u = AtomicNode("1")
        v = AtomicNode("2")
        self.assertFalse(u <= v)

    def test_schemaNodeLT_betweenAtomicNodes_returnsFalseIfNodesEqual(self):
        u = AtomicNode("1")
        v = AtomicNode("1")
        self.assertFalse(u < v)

    def test_schemaNodeLT_betweenAtomicNodes_returnsFalseIfNodesNotEqual(self):
        u = AtomicNode("1")
        v = AtomicNode("2")
        self.assertFalse(u < v)

    def test_schemaNodeLTEQ_betweenProductNodes_isReflexive(self):
        u = AtomicNode("1")
        v = AtomicNode("2")
        p1 = SchemaNode.product([u, v])
        p2 = SchemaNode.product([u, v])
        self.assertTrue(p1 <= p2)

    def test_schemaNodeLTEQ_betweenProductNodes_returnsTrueIfOneNodeSubsetOfAnother(self):
        u = AtomicNode("1")
        v = AtomicNode("2")
        w = AtomicNode("3")
        p1 = SchemaNode.product([u, v])
        p2 = SchemaNode.product([u, v, w])
        self.assertTrue(p1 <= p2)

    def test_schemaNodeLTEQ_betweenProductNodes_returnsFalseIfOneNodeNotSubsetOfAnother(self):
        u = AtomicNode("1")
        v = AtomicNode("2")
        w = AtomicNode("3")
        x = AtomicNode("4")
        p1 = SchemaNode.product([u, v, x])
        p2 = SchemaNode.product([u, v, w])
        self.assertFalse(p1 <= p2)

    def test_schemaNodeProduct_isAssociative(self):
        u = AtomicNode("u")
        v = AtomicNode("v")
        w = AtomicNode("a")
        u.id_prefix = 0
        v.id_prefix = 0
        w.id_prefix = 0
        p1 = SchemaNode.product([u, v])
        p2 = SchemaNode.product([p1, w])
        p3 = SchemaNode.product([v, w])
        p4 = SchemaNode.product([u, p3])
        self.assertExpectedInline(str(p2), """u;v;a""")
        self.assertEqual(p2, p4)

    def test_getConstituents_returnsSingleton_ifNodeAtomic(self):
        u = AtomicNode("u")
        u.id_prefix = 0
        self.assertExpectedInline(str(SchemaNode.get_constituents(u)), """[u]""")

    def test_getConstituents_returnsSet_ifNodeNotAtomic(self):
        u = AtomicNode("u")
        u.id_prefix = 0
        v = AtomicNode("v")
        v.id_prefix = 0
        p = SchemaNode.product([u, v])
        self.assertExpectedInline(str(SchemaNode.get_constituents(p)), """[u, v]""")

    def test_isEquivalent_onAtomicNodes_returnsTrueIfSameEquivalenceClass(self):
        u = AtomicNode("u")
        v = AtomicNode("v")
        e = UnionFind.initialise()
        e = UnionFind.add_singletons(e, [u, v])
        e = UnionFind.union(e, u, v)
        self.assertTrue(SchemaNode.is_equivalent(u, v, e))

    def test_isEquivalent_onProductNodes_actsElementWise(self):
        u1 = AtomicNode("u1")
        v1 = AtomicNode("v1")
        u2 = AtomicNode("u2")
        v2 = AtomicNode("v2")
        e = UnionFind.initialise()
        e = UnionFind.add_singletons(e, [u1, v1, u2, v2])
        e = UnionFind.union(e, u1, v1)
        e = UnionFind.union(e, u2, v2)
        p1 = SchemaNode.product([u1, u2])
        p2 = SchemaNode.product([v1, v2])
        self.assertTrue(SchemaNode.is_equivalent(p1, p2, e))




