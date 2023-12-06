import expecttest

from schema import SchemaNode, SchemaNodeNameShouldNotContainSemicolonException
from union_find.union_find import UnionFind


class TestSchemaNode(expecttest.TestCase):
    def test_constructSchemaNode_raisesException_ifSchemaNodeNameContainsSemicolon(self):
        self.assertExpectedRaisesInline(SchemaNodeNameShouldNotContainSemicolonException, lambda: SchemaNode("hello; goodbye"), """Schema node name should not contain a semicolon. Name: hello; goodbye""")

    def test_schemaNodeEquality_returnsTrue_ifNodesAtomicAndNameEqual(self):
        u = SchemaNode("name")
        v = SchemaNode("name")
        self.assertEqual(u, v)

    def test_schemaNodeEquality_returnsFalse_ifNodesAtomicAndNameNotEqual(self):
        u = SchemaNode("name")
        v = SchemaNode("name'")
        self.assertNotEqual(u, v)

    def test_schemaNodeEquality_returnsFalse_whenComparingAtomicNodeWithProductNode(self):
        u = SchemaNode("1")
        v = SchemaNode("2")
        w = SchemaNode.product([u, v])
        self.assertNotEqual(u, w)

    def test_schemaNodeEquality_returnsTrue_whenComparingProductNodes(self):
        u = SchemaNode("1")
        v = SchemaNode("2")
        w1 = SchemaNode.product([u, v])
        w2 = SchemaNode.product([v, u])
        self.assertEqual(w1, w2)

    def test_schemaNodeEquality_returnsFalse_whenComparingUnequalProductNodes(self):
        u = SchemaNode("1")
        v = SchemaNode("2")
        w = SchemaNode("3")
        w1 = SchemaNode.product([u, v])
        w2 = SchemaNode.product([v, w])
        self.assertNotEqual(w1, w2)

    def test_schemaNodeProduct_returnsOriginalNode_whenGivenOneNode(self):
        u = SchemaNode("1")
        w = SchemaNode.product([u])
        self.assertEqual(u, w)

    def test_schemaNodeLTEQ_betweenAtomicNodes_returnsTrueIfNodesEqual(self):
        u = SchemaNode("1")
        v = SchemaNode("1")
        self.assertTrue(u <= v)

    def test_schemaNodeLTEQ_betweenAtomicNodes_returnsFalseIfNodesNotEqual(self):
        u = SchemaNode("1")
        v = SchemaNode("2")
        self.assertFalse(u <= v)

    def test_schemaNodeLT_betweenAtomicNodes_returnsFalseIfNodesEqual(self):
        u = SchemaNode("1")
        v = SchemaNode("1")
        self.assertFalse(u < v)

    def test_schemaNodeLT_betweenAtomicNodes_returnsFalseIfNodesNotEqual(self):
        u = SchemaNode("1")
        v = SchemaNode("2")
        self.assertFalse(u < v)

    def test_schemaNodeLTEQ_betweenProductNodes_isReflexive(self):
        u = SchemaNode("1")
        v = SchemaNode("2")
        p1 = SchemaNode.product([u, v])
        p2 = SchemaNode.product([u, v])
        self.assertTrue(p1 <= p2)

    def test_schemaNodeLTEQ_betweenProductNodes_returnsTrueIfOneNodeSubsetOfAnother(self):
        u = SchemaNode("1")
        v = SchemaNode("2")
        w = SchemaNode("3")
        p1 = SchemaNode.product([u, v])
        p2 = SchemaNode.product([u, v, w])
        self.assertTrue(p1 <= p2)

    def test_schemaNodeLTEQ_betweenProductNodes_returnsFalseIfOneNodeNotSubsetOfAnother(self):
        u = SchemaNode("1")
        v = SchemaNode("2")
        w = SchemaNode("3")
        x = SchemaNode("4")
        p1 = SchemaNode.product([u, v, x])
        p2 = SchemaNode.product([u, v, w])
        self.assertFalse(p1 <= p2)

    def test_schemaNodeProduct_maintainsCluster_ifAllConstituentsFromSameCluster(self):
        u = SchemaNode("u", cluster="1")
        v = SchemaNode("v", cluster="1")
        w = SchemaNode("a", cluster="1")
        p = SchemaNode.product([u, v, w])
        self.assertExpectedInline(str(p), """1.u;v;a""")

    def test_schemaNodeProduct_isAssociative(self):
        u = SchemaNode("u", cluster="1")
        v = SchemaNode("v", cluster="1")
        w = SchemaNode("a", cluster="1")
        p1 = SchemaNode.product([u, v])
        p2 = SchemaNode.product([p1, w])
        p3 = SchemaNode.product([v, w])
        p4 = SchemaNode.product([p3, u])
        self.assertExpectedInline(str(p2), """1.u;v;a""")
        self.assertEqual(p2, p4)

    def test_getConstituents_returnsSingleton_ifNodeAtomic(self):
        u = SchemaNode("u", cluster="1")
        self.assertExpectedInline(str(SchemaNode.get_constituents(u)), """[1.u]""")

    def test_getConstituents_returnsSet_ifNodeNotAtomic(self):
        u = SchemaNode("u", cluster="1")
        v = SchemaNode("v", cluster="2")
        p = SchemaNode.product([u, v])
        self.assertExpectedInline(str(SchemaNode.get_constituents(p)), """[1.u, 2.v]""")

    def test_isEquivalent_onAtomicNodes_returnsTrueIfSameEquivalenceClass(self):
        u = SchemaNode("u", cluster="1")
        v = SchemaNode("v", cluster="2")
        e = UnionFind.initialise()
        e = UnionFind.add_singletons(e, [u, v])
        e = UnionFind.union(e, u, v)
        self.assertTrue(SchemaNode.is_equivalent(u, v, e))

    def test_isEquivalent_onProductNodes_actsElementWise(self):
        u1 = SchemaNode("u1", cluster="1")
        v1 = SchemaNode("v1", cluster="2")
        u2 = SchemaNode("u2", cluster="1")
        v2 = SchemaNode("v2", cluster="2")
        e = UnionFind.initialise()
        e = UnionFind.add_singletons(e, [u1, v1, u2, v2])
        e = UnionFind.union(e, u1, v1)
        e = UnionFind.union(e, u2, v2)
        p1 = SchemaNode.product([u1, u2])
        p2 = SchemaNode.product([v1, v2])
        self.assertTrue(SchemaNode.is_equivalent(p1, p2, e))




