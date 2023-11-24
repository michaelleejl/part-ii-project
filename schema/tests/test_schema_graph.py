import unittest

import numpy as np
import pandas as pd

from schema import SchemaGraph, SchemaNode, SchemaEdge
from schema.cardinality import Cardinality


class TestSchemaGraph(unittest.TestCase):
    def test_schemaGraph_addingRelationIsIdentify_ifFromAndToNodeEqual(self):
        g = SchemaGraph()
        u = SchemaNode("name", pd.DataFrame([0, 1, 2]), family="1")
        e = SchemaEdge(u, u, pd.DataFrame({"name": [0, 1, 2]}))
        g.add_relation(e)
        self.assertEqual(0, len(g.adjacencyList.keys()))

    def test_schemaGraph_addingRelationSucceeds_ifFromAndToNodeNotEqual(self):
        g = SchemaGraph()
        u = SchemaNode("name", pd.DataFrame([0, 1, 2]), family="1")
        v = SchemaNode("name2", pd.DataFrame([3, 2, 1]), family="1")
        mapping = pd.DataFrame({"name": [0, 1, 2], "name2": [3, 2, 1]})
        e = SchemaEdge(u, v, mapping)
        g.add_relation(e)
        self.assertEqual(2, len(g.adjacencyList.keys()))
        self.assertTrue(u in g.adjacencyList.keys())
        self.assertEqual(1, len(g.adjacencyList[u]))
        self.assertEqual(e, g.adjacencyList[u][0])
        self.assertEqual(Cardinality.ONE_TO_ONE, g.adjacencyList[u][0].get_cardinality(u))
        self.assertTrue(np.all(mapping == g.adjacencyList[u][0].mapping))
        self.assertTrue(v in g.adjacencyList.keys())
        self.assertEqual(1, len(g.adjacencyList[v]))
        self.assertEqual(e, g.adjacencyList[v][0])
        self.assertEqual(Cardinality.ONE_TO_ONE, g.adjacencyList[v][0].get_cardinality(v))
        self.assertTrue(np.all(mapping == g.adjacencyList[v][0].mapping))
