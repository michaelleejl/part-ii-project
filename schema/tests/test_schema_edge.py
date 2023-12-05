import unittest

import pandas as pd

from schema import SchemaNode, SchemaEdge, Cardinality


class TestSchemaEdge(unittest.TestCase):

    def test_schemaEdge_getCardinalityReturnsCorrectCardinalityIfTraversingForwards(self):
        u = SchemaNode("name")
        v = SchemaNode("name2")
        e = SchemaEdge(u, v, Cardinality.MANY_TO_ONE)
        self.assertEqual(Cardinality.MANY_TO_ONE, e.get_cardinality(u))

    def test_schemaEdge_getCardinalityReturnsCorrectCardinalityIfTraversingBackwards(self):
        u = SchemaNode("name")
        v = SchemaNode("name2")
        e = SchemaEdge(u, v, Cardinality.MANY_TO_ONE)
        self.assertEqual(Cardinality.ONE_TO_MANY, e.get_cardinality(v))