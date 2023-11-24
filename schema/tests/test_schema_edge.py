import unittest

import pandas as pd

from schema import SchemaNode, SchemaEdge, Cardinality


class TestSchemaEdge(unittest.TestCase):

    def test_schemaEdge_getCardinalityReturnsCorrectCardinalityIfTraversingForwards(self):
        u = SchemaNode("name", pd.DataFrame({"name": [0, 1, 2]}), family="1")
        v = SchemaNode("name2", pd.DataFrame({"name2": [3, 4, 5]}), family="1")
        mapping = pd.DataFrame({f"{hash(u)}_name": [0, 1], f"{hash(v)}_name2": [4, 4]})
        e = SchemaEdge(u, v, mapping)
        self.assertEqual(Cardinality.MANY_TO_ONE, e.get_cardinality(u))

    def test_schemaEdge_getCardinalityReturnsCorrectCardinalityIfTraversingBackwards(self):
        u = SchemaNode("name", pd.DataFrame({"name": [0, 1, 2]}), family="1")
        v = SchemaNode("name2", pd.DataFrame({"name2": [3, 4, 5]}), family="1")
        mapping = pd.DataFrame({f"{hash(u)}_name": [0, 1], f"{hash(v)}_name2": [4, 4]})
        e = SchemaEdge(u, v, mapping)
        self.assertEqual(Cardinality.ONE_TO_MANY, e.get_cardinality(v))

    def test_schemaEdge_extendsMappingIfNewMappingDoesNotConflictWithOldMapping(self):
        u = SchemaNode("name", pd.DataFrame({"name": [0, 1, 2]}), family="1")
        v = SchemaNode("name2", pd.DataFrame({"name2": [3, 4, 5]}), family="1")
        print(u.data)
        mapping = pd.DataFrame({f"{hash(u)}_name": [0, 1], f"{hash(v)}_name2": [3, 4]})
        e1 = SchemaEdge(u, v, mapping)
        new_mapping = pd.DataFrame({f"{hash(u)}_name": [1, 2], f"{hash(v)}_name2": [4, 5]})
        e2 = SchemaEdge.extend_mapping(e1, new_mapping)
        expected = pd.DataFrame({f"{hash(u)}_name": [0, 1, 2], f"{hash(v)}_name2": [3, 4, 5]}).astype(object)
        self.assertTrue((expected.equals(e2.mapping)))

    def test_schemaEdge_updatesMappingEvenIfNewMappingConflictsWithOldMappings(self):
        u = SchemaNode("name", pd.DataFrame({"name": [0, 1, 2]}), family="1")
        v = SchemaNode("name2", pd.DataFrame({"name2": [3, 4, 5]}), family="1")
        mapping = pd.DataFrame({f"{hash(u)}_name": [0, 1], f"{hash(v)}_name2": [3, 4]})
        e1 = SchemaEdge(u, v, mapping)
        new_mapping = pd.DataFrame({f"{hash(u)}_name": [1, 2], f"{hash(v)}_name2": [5, 5]})
        e2 = SchemaEdge.update_mapping(e1, new_mapping)
        expected = pd.DataFrame({f"{hash(u)}_name": [0, 1, 2], f"{hash(v)}_name2": [3, 5, 5]}).astype(object)
        self.assertTrue((expected.equals(e2.mapping)))

    def test_schemaEdgeEquality_returnsTrue_ifEndPointsEqual(self):
        u = SchemaNode("name", pd.DataFrame([0, 1, 2]), family="1")
        v = SchemaNode("name2", pd.DataFrame([3, 2, 1]), family="1")
        e1 = SchemaEdge(u, v, pd.DataFrame({"name": [0, 1, 2], "name2": [3, 2, 1]}))
        e2 = SchemaEdge(u, v, pd.DataFrame({"name": [0, 1, 2], "name2": [1, 2, 3]}))
        self.assertEqual(e1, e2)

    def test_schemaEdgeEquality_returnsFalse_ifEndPointsNotEqual(self):
        u = SchemaNode("name", pd.DataFrame([0, 1, 2]), family="1")
        v = SchemaNode("name2", pd.DataFrame([3, 2, 1]), family="1")
        e1 = SchemaEdge(u, v, pd.DataFrame({"name": [0, 1, 2], "name2": [3, 2, 1]}))
        e2 = SchemaEdge(v, u, pd.DataFrame({"name": [0, 1, 2], "name2": [1, 2, 3]}))
        self.assertNotEqual(e1, e2)