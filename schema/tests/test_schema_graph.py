import expecttest
import numpy as np
import pandas as pd

from schema import SchemaGraph, SchemaNode, SchemaEdge
from schema.cardinality import Cardinality


class TestSchemaGraph(expecttest.TestCase):

    def test_schemaGraph_addNodeIsIdempotent(self):
        g = SchemaGraph()
        u = SchemaNode("name", cluster="1")
        g.add_node(u)
        g.add_node(u)
        self.assertExpectedInline(str(g.schema_nodes), """[1.(name)]""")
        self.assertExpectedInline(str(g), """\
                                                    FULLY CONNECTED CLUSTERS 
                                                    ==========================
                                                    
                                                    
                                                    ==========================
                                                    
                                                    ADJACENCY LIST 
                                                    ==========================
                                                    
                                                    
                                                    ==========================
                                                    
                                                    EQUIVALENCE CLASSES 
                                                    ==========================
                                                    
                                                    ==========================
                                                    Class 0
                                                    --------------------------
                                                    1.(name)
                                                    ==========================
                                                    
                                                    ==========================
                                                    """)

    def test_schemaGraph_addNodesActsLikeSetUnion(self):
        g = SchemaGraph()
        u = SchemaNode("name", cluster="1")
        v = SchemaNode("name2", cluster="1")
        w = SchemaNode("name3", cluster="1")
        g.add_nodes([u, v])
        g.add_nodes([v, w])
        self.assertExpectedInline(str(g.schema_nodes), """[1.(name), 1.(name2), 1.(name3)]""")
        self.assertExpectedInline(str(g), """\
                                                    FULLY CONNECTED CLUSTERS 
                                                    ==========================
                                                    
                                                    
                                                    ==========================
                                                    
                                                    ADJACENCY LIST 
                                                    ==========================
                                                    
                                                    
                                                    ==========================
                                                    
                                                    EQUIVALENCE CLASSES 
                                                    ==========================
                                                    
                                                    ==========================
                                                    Class 0
                                                    --------------------------
                                                    1.(name)
                                                    ==========================
                                                    
                                                    ==========================
                                                    Class 1
                                                    --------------------------
                                                    1.(name2)
                                                    ==========================
                                                    
                                                    ==========================
                                                    Class 2
                                                    --------------------------
                                                    1.(name3)
                                                    ==========================
                                                    
                                                    ==========================
                                                    """)

    def test_schemaGraph_blendNodesMergesEquivalenceClasses(self):
        g = SchemaGraph()
        u = SchemaNode("name", cluster="1")
        v = SchemaNode("name2", cluster="1")
        g.add_nodes([u, v])
        g.blend_nodes(u, v)
        self.assertExpectedInline(str(g), """\
                                                    FULLY CONNECTED CLUSTERS 
                                                    ==========================
                                                    
                                                    
                                                    ==========================
                                                    
                                                    ADJACENCY LIST 
                                                    ==========================
                                                    
                                                    
                                                    ==========================
                                                    
                                                    EQUIVALENCE CLASSES 
                                                    ==========================
                                                    
                                                    ==========================
                                                    Class 0
                                                    --------------------------
                                                    1.(name)
                                                    1.(name2)
                                                    ==========================
                                                    
                                                    ==========================
                                                    """)

    def test_schemaGraph_addingRelationSucceeds_ifFromAndToNodeNotEqual(self):
        g = SchemaGraph()
        u = SchemaNode("name", pd.DataFrame([0, 1, 2]), family="1")
        v = SchemaNode("name2", pd.DataFrame([3, 2, 1]), family="1")
        mapping = pd.DataFrame({"name": [0, 1, 2], "name2": [3, 2, 1]})
        e = SchemaEdge(u, v, mapping)
        g.add_edge(e)
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
