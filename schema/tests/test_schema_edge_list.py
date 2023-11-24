import unittest

import numpy as np
import pandas as pd

from schema import SchemaNode, SchemaEdge, Cardinality, SchemaEdgeList, EdgeAlreadyExistsException


class TestSchemaEdgeList(unittest.TestCase):
    def test_schemaEdgeList_addingToEdgeListSucceedsIfEdgeDoesNotAlreadyExist(self):
        u = SchemaNode("name", pd.DataFrame([0, 1, 2]), family="1")
        v = SchemaNode("name2", pd.DataFrame([3, 2, 1]), family="1")
        e = SchemaEdge(u, v, pd.DataFrame({f"{hash(u)}_name": [0, 1, 2], f"{hash(v)}_name2": [3, 2, 1]}))
        es = SchemaEdgeList(frozenset([e]))
        updated = pd.DataFrame({f"{hash(u)}_name": [0, 1, 2], f"{hash(v)}_name2": [1, 2, 3]})
        e2 = SchemaEdge(v, u, updated)
        esl = SchemaEdgeList.add_edge(es, e2)
        self.assertEqual(2, len(esl))

    def test_schemaEdgeList_addingToEdgeListRaisesExceptionIfEdgeAlreadyExists(self):
        u = SchemaNode("name", pd.DataFrame([0, 1, 2]), family="1")
        v = SchemaNode("name2", pd.DataFrame([3, 2, 1]), family="1")
        e = SchemaEdge(u, v, pd.DataFrame({f"{hash(u)}_name": [0, 1, 2], f"{hash(v)}_name2": [3, 2, 1]}))
        es = SchemaEdgeList(frozenset([e]))
        updated = pd.DataFrame({f"{hash(u)}_name": [0, 1, 2], f"{hash(v)}_name2": [1, 2, 3]})
        e2 = SchemaEdge(u, v, updated)
        self.assertRaises(EdgeAlreadyExistsException, lambda: SchemaEdgeList.add_edge(es, e2))

    def test_schemaEdgeList_updatingEdgeListSucceeds(self):
        u = SchemaNode("name", pd.DataFrame([0, 1, 2]), family="1")
        v = SchemaNode("name2", pd.DataFrame([3, 2, 1]), family="1")
        e = SchemaEdge(u, v, pd.DataFrame({f"{hash(u)}_name": [0, 1, 2], f"{hash(v)}_name2": [3, 2, 1]}))
        es = SchemaEdgeList(frozenset([e]))
        updated = pd.DataFrame({f"{hash(u)}_name": [0, 0, 2], f"{hash(v)}_name2": [1, 2, 1]})
        e2 = SchemaEdge(u, v, updated)
        esl = SchemaEdgeList.replace_edge(es, e2).get_edge_list()
        self.assertEqual(1, len(esl))
        self.assertEqual(Cardinality.MANY_TO_MANY, esl[0].cardinality)
        self.assertTrue(np.all(updated == esl[0].mapping))