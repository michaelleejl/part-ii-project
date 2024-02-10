import unittest


from schema import SchemaNode, SchemaEdge, Cardinality, AtomicNode, BaseType


class TestSchemaEdge(unittest.TestCase):

    def test_schemaEdge_getCardinalityReturnsCorrectCardinalityIfTraversingForwards(
        self,
    ):
        u = AtomicNode("name", BaseType.STRING)
        v = AtomicNode("name2", BaseType.STRING)
        e = SchemaEdge(u, v, Cardinality.MANY_TO_ONE)
        self.assertEqual(Cardinality.MANY_TO_ONE, e.get_cardinality(u))

    def test_schemaEdge_getCardinalityReturnsCorrectCardinalityIfTraversingBackwards(
        self,
    ):
        u = AtomicNode("name", BaseType.STRING)
        v = AtomicNode("name2", BaseType.STRING)
        e = SchemaEdge(u, v, Cardinality.MANY_TO_ONE)
        self.assertEqual(Cardinality.ONE_TO_MANY, e.get_cardinality(v))
