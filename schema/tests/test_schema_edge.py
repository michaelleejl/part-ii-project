import expecttest


from schema.node import SchemaNode, AtomicNode, BaseType
from schema.cardinality import Cardinality
from schema.edge import SchemaEdge


class TestSchemaEdge(expecttest.TestCase):

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

