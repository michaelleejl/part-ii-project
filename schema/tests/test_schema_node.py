import unittest

from schema import SchemaNode, SchemaNodeNameShouldNotContainSemicolonException, SchemaGraph


class TestSchemaNode(unittest.TestCase):
    def test_constructSchemaNode_raisesException_ifSchemaNodeNameContainsSemicolon(self):
        self.assertRaises(SchemaNodeNameShouldNotContainSemicolonException, lambda: SchemaNode("hello; goodbye"))

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
        w = SchemaNode.product(frozenset([u, v]))
        self.assertNotEqual(u, w)

    def test_schemaNodeEquality_returnsTrue_whenComparingProductNodes(self):
        u = SchemaNode("1")
        v = SchemaNode("2")
        w1 = SchemaNode.product(frozenset([u, v]))
        w2 = SchemaNode.product(frozenset([v, u]))
        self.assertEqual(w1, w2)

    def test_schemaNodeEquality_returnsFalse_whenComparingUnequalProductNodes(self):
        u = SchemaNode("1")
        v = SchemaNode("2")
        w = SchemaNode("3")
        w1 = SchemaNode.product(frozenset([u, v]))
        w2 = SchemaNode.product(frozenset([v, w]))
        self.assertNotEqual(w1, w2)

    def test_schemaNodeProduct_returnsOriginalNode_whenGivenOneNode(self):
        u = SchemaNode("1")
        w = SchemaNode.product(frozenset([u]))
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
        p1 = SchemaNode.product(frozenset([u, v]))
        p2 = SchemaNode.product(frozenset([u, v]))
        self.assertTrue(p1 <= p2)

    def test_schemaNodeLTEQ_betweenProductNodes_returnsTrueIfOneNodeSubsetOfAnother(self):
        u = SchemaNode("1")
        v = SchemaNode("2")
        w = SchemaNode("3")
        p1 = SchemaNode.product(frozenset([u, v]))
        p2 = SchemaNode.product(frozenset([u, v, w]))
        self.assertTrue(p1 <= p2)

    def test_schemaNodeLTEQ_betweenProductNodes_returnsFalseIfOneNodeNotSubsetOfAnother(self):
        u = SchemaNode("1")
        v = SchemaNode("2")
        w = SchemaNode("3")
        x = SchemaNode("4")
        p1 = SchemaNode.product(frozenset([u, v, x]))
        p2 = SchemaNode.product(frozenset([u, v, w]))
        self.assertFalse(p1 <= p2)
