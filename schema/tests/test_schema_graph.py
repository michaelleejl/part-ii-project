import expecttest
import numpy as np
import pandas as pd

from schema import SchemaGraph, SchemaNode, SchemaEdge, AllNodesInClusterMustAlreadyBeInGraphException, \
    AllNodesInFullyConnectedClusterMustHaveSameClusterException, NodeNotInSchemaGraphException, \
    FindingEdgeViaNodeMustRespectEquivalence, Transform
from schema.cardinality import Cardinality


class TestSchemaGraph(expecttest.TestCase):

    def test_schemaGraph_addNodeIsIdempotent(self):
        g = SchemaGraph()
        u = SchemaNode("name", cluster="1")
        g.add_node(u)
        g.add_node(u)
        self.assertExpectedInline(str(g.schema_nodes), """[1.name]""")
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
1.name
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
        self.assertExpectedInline(str(g.schema_nodes), """[1.name, 1.name2, 1.name3]""")
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
1.name
==========================

==========================
Class 1
--------------------------
1.name2
==========================

==========================
Class 2
--------------------------
1.name3
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
1.name
1.name2
==========================

==========================
""")

    def test_schemaGraph_areNodesEqual_returnsTrue_ifNodesInSameEquivalenceClass(self):
        g = SchemaGraph()
        u = SchemaNode("name", cluster="1")
        v = SchemaNode("name2", cluster="1")
        g.add_nodes([u, v])
        g.blend_nodes(u, v)
        self.assertTrue(g.are_nodes_equal(u, v))

    def test_schemaGraph_areNodesEqual_returnsFalse_ifNodesInDifferentEquivalenceClasses(self):
        g = SchemaGraph()
        u = SchemaNode("name", cluster="1")
        v = SchemaNode("name2", cluster="1")
        g.add_nodes([u, v])
        self.assertFalse(g.are_nodes_equal(u, v))

    def test_schemaGraph_addFullyConnectedCluster_raisesExceptionIfNodesNotInGraph(self):
        g = SchemaGraph()
        u = SchemaNode("name", cluster="1")
        v = SchemaNode("name2", cluster="1")
        w = SchemaNode("name3", cluster="1")
        g.add_nodes([u, v])
        self.assertExpectedRaisesInline(AllNodesInClusterMustAlreadyBeInGraphException,
                                        lambda: g.add_fully_connected_cluster([u, v, w], u),
                                        """When adding a cluster, all nodes in the cluster must already exist in the graph. The following nodes are not in the graph: 1.name3""")

    def test_schemaGraph_addFullyConnectedCluster_raisesExceptionIfNodesDoNotShareACluster(self):
        g = SchemaGraph()
        u = SchemaNode("name", cluster="1")
        v = SchemaNode("name2", cluster="2")
        g.add_nodes([u, v])
        self.assertExpectedRaisesInline(AllNodesInFullyConnectedClusterMustHaveSameClusterException,
                                        lambda: g.add_fully_connected_cluster([u, v], u),
                                        """All nodes in a fully connected cluster must have the same cluster attribute""")

    def test_schemaGraph_addFullyConnectedCluster_succeeds(self):
        g = SchemaGraph()
        u = SchemaNode("name", cluster="1")
        v = SchemaNode("name2", cluster="1")
        g.add_nodes([u, v])
        g.add_fully_connected_cluster([u, v], u)
        self.assertExpectedInline(str(g), """\
FULLY CONNECTED CLUSTERS 
==========================

==========================
1
--------------------------
1.name
1.name2
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
1.name
==========================

==========================
Class 1
--------------------------
1.name2
==========================

==========================
""")

    def test_schemaGraph_addEdgeRaisesException_ifFromNodeNotInGraph(self):
        g = SchemaGraph()
        u = SchemaNode("u", cluster="1")
        v = SchemaNode("v", cluster="1")
        self.assertExpectedRaisesInline(NodeNotInSchemaGraphException, lambda: g.add_edge(u, v),
                                        """Node 1.u is not in schema graph.""")

    def test_schemaGraph_addEdgeRaisesException_ifToNodeNotInGraph(self):
        g = SchemaGraph()
        u = SchemaNode("u", cluster="1")
        v = SchemaNode("v", cluster="1")
        g.add_node(u)
        self.assertExpectedRaisesInline(NodeNotInSchemaGraphException, lambda: g.add_edge(u, v),
                                        """Node 1.v is not in schema graph.""")

    def test_schemaGraph_addEdgeDoesNothing_ifFromNodeEqualsToNode(self):
        g = SchemaGraph()
        u = SchemaNode("u", cluster="1")
        g.add_node(u)
        g.add_edge(u, u)
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
1.u
==========================

==========================
""")

    def test_schemaGraphAddEdge_successfullyAddsNewEdgeToAdjacencyList(self):
        g = SchemaGraph()
        u = SchemaNode("u", cluster="1")
        v = SchemaNode("v", cluster="2")
        g.add_node(u)
        g.add_node(v)
        g.add_edge(u, v, Cardinality.ONE_TO_ONE)
        self.assertExpectedInline(str(g), """\
FULLY CONNECTED CLUSTERS 
==========================


==========================

ADJACENCY LIST 
==========================

==========================
1.u
--------------------------
u <--> v
==========================

==========================
2.v
--------------------------
u <--> v
==========================

==========================

EQUIVALENCE CLASSES 
==========================

==========================
Class 0
--------------------------
1.u
==========================

==========================
Class 1
--------------------------
2.v
==========================

==========================
""")

    def test_schemaGraph_getDirectEdgeBetweenNodes_passesEquivalenceCheckIfNodesAreEquivalent(self):
        g = SchemaGraph()
        u = SchemaNode("u", cluster='1')
        v = SchemaNode("v", cluster="2")
        w = SchemaNode("w", cluster="3")
        g.add_nodes([u, v, w])
        g.blend_nodes(u, v)
        g.add_edge(v, w)
        e = g.get_direct_edge_between_nodes(u, w, via=v)
        self.assertExpectedInline(str(e), """(True, v --- w)""")

    def test_schemaGraph_getDirectEdgeBetweenNodes_failsEquivalenceCheckIfNodesAreNotEquivalent(self):
        g = SchemaGraph()
        u = SchemaNode("u", cluster='1')
        v = SchemaNode("v", cluster="2")
        w = SchemaNode("w", cluster="3")
        g.add_nodes([u, v, w])
        g.add_edge(v, w)
        self.assertExpectedRaisesInline(FindingEdgeViaNodeMustRespectEquivalence, lambda: g.get_direct_edge_between_nodes(u, w, via=v),
                                        """When finding an edge between node1 and node2 via node3, node1 and node3 must be equivalent. 1.u and 2.v are not equivalent.""")

    def test_schemaGraph_getDirectEdgeBetweenNodes_findsIdentityRelationBetweenNodeAndItself(self):
        g = SchemaGraph()
        u = SchemaNode("u", cluster="1")
        g.add_node(u)
        self.assertExpectedInline(str(g.get_direct_edge_between_nodes(u, u)), """(True, u <--> u)""")

    def test_schemaGraph_getDirectEdgeBetweenNodes_findsProjectionIfEndNodeProjectionOfStartNode(self):
        g = SchemaGraph()
        u = SchemaNode("u", cluster="1")
        v = SchemaNode("v", cluster="2")
        p = SchemaNode.product([u, v])
        g.add_nodes([u, v])
        self.assertExpectedInline(str(g.get_direct_edge_between_nodes(p, u)),
                                  """(True, u;v ---> u)""")

    def test_schemaGraph_getDirectEdgeBetweenNodes_findsExpansionIfStartNodeProjectionOfEndNode(self):
        g = SchemaGraph()
        u = SchemaNode("u", cluster="1")
        v = SchemaNode("v", cluster="2")
        p = SchemaNode.product([u, v])
        g.add_nodes([u, v])
        self.assertExpectedInline(str(g.get_direct_edge_between_nodes(u, p)),
                                  """(True, u <--- u;v)""")

    def test_schemaGraph_getDirectEdgeBetweenNodes_findsBijectionIfNodesAreEquivalent(self):
        g = SchemaGraph()
        u = SchemaNode("u", cluster="1")
        v = SchemaNode("v", cluster="2")
        g.add_nodes([u, v])
        g.blend_nodes(u, v)
        self.assertExpectedInline(str(g.get_direct_edge_between_nodes(u, v)),
                                  """(True, u <--> v)""")

    def test_schemaGraph_getDirectEdgeBetweenNodes_findsManyToOneRelation_ifNodesAreInClusterAndStartNodeIsKey(self):
        g = SchemaGraph()
        u = SchemaNode("u", cluster="1")
        v = SchemaNode("v", cluster="1")
        w = SchemaNode("w", cluster="1")
        g.add_nodes([u, v, w])
        p = SchemaNode.product([u, v])
        g.add_fully_connected_cluster([u, v, w], p)
        e = g.get_direct_edge_between_nodes(p, w)
        self.assertExpectedInline(str(e), """(True, u;v ---> w)""")

    def test_schemaGraph_getDirectEdgeBetweenNodes_findsOneToManyRelation_ifNodesAreInClusterAndEndNodeIsKey(self):
        g = SchemaGraph()
        u = SchemaNode("u", cluster="1")
        v = SchemaNode("v", cluster="1")
        w = SchemaNode("w", cluster="1")
        g.add_nodes([u, v, w])
        p = SchemaNode.product([u, v])
        g.add_fully_connected_cluster([u, v, w], p)
        e = g.get_direct_edge_between_nodes(w, p)
        self.assertExpectedInline(str(e), """(True, w <--- u;v)""")

    def test_schemaGraph_getDirectEdgeBetweenNodes_findsManyToManyRelation_ifNodesAreInClusterAndStartAndEndNodesNotKey(self):
        g = SchemaGraph()
        u = SchemaNode("u", cluster="1")
        v = SchemaNode("v", cluster="1")
        w = SchemaNode("w", cluster="1")
        g.add_nodes([u, v, w])
        p = SchemaNode.product([u, v])
        g.add_fully_connected_cluster([u, v, w], p)
        e = g.get_direct_edge_between_nodes(u, w)
        self.assertExpectedInline(str(e), """(True, u --- w)""")

    def test_schemaGraph_getDirectEdgeBetweenNodes_findsEdge_ifInAdjacencyList(self):
        g = SchemaGraph()
        u = SchemaNode("u", cluster="1")
        v = SchemaNode("v", cluster="1")
        g.add_nodes([u, v])
        g.add_edge(u, v)
        e = g.get_direct_edge_between_nodes(u, v)
        self.assertExpectedInline(str(e), """(True, u --- v)""")

    def test_schemaGraph_getEdgeBetweenNodes_actsElementwise_andCorrectlyComputesManyToManyCardinality(self):
        g = SchemaGraph()
        u = SchemaNode("u", cluster="1")
        v = SchemaNode("v", cluster="1")
        w = SchemaNode("w", cluster="1")
        x = SchemaNode("x", cluster="2")
        y = SchemaNode("y", cluster="3")
        z = SchemaNode("z", cluster="4")
        g.add_nodes([u, v, w, x, y, z])
        p = SchemaNode.product([u, v])
        g.add_fully_connected_cluster([u, v, w], p)
        g.blend_nodes(u, x)
        g.add_edge(y, z)
        e = g.get_edge_between_nodes([
            Transform(SchemaNode.product([x, v]), w, p),
            Transform(y, z)
        ])
        self.assertExpectedInline(str(e), """(True, x;v;y --- w;z)""")

    def test_schemaGraph_getEdgeBetweenNodes_actsElementwise_andCorrectlyComputesManyToOneCardinality(self):
        g = SchemaGraph()
        u = SchemaNode("u", cluster="1")
        v = SchemaNode("v", cluster="1")
        w = SchemaNode("w", cluster="1")
        x = SchemaNode("x", cluster="2")
        y = SchemaNode("y", cluster="3")
        g.add_nodes([u, v, w, x, y])
        p = SchemaNode.product([u, v])
        g.add_fully_connected_cluster([u, v, w], p)
        g.blend_nodes(u, x)
        g.add_edge(x, y, Cardinality.MANY_TO_ONE)
        e = g.get_edge_between_nodes([
            Transform(SchemaNode.product([x, v]), w, p),
            Transform(x, y)
        ])
        self.assertExpectedInline(str(e), """(True, x;v ---> w;y)""")

    def test_schemaGraph_getEdgeBetweenNodes_failsIfAnyTransformDoesNotExist(self):
        g = SchemaGraph()
        u = SchemaNode("u", cluster="1")
        v = SchemaNode("v", cluster="1")
        w = SchemaNode("w", cluster="1")
        x = SchemaNode("x", cluster="2")
        y = SchemaNode("y", cluster="3")
        g.add_nodes([u, v, w, x, y])
        p = SchemaNode.product([u, v])
        g.add_fully_connected_cluster([u, v, w], p)
        g.blend_nodes(u, x)
        e = g.get_edge_between_nodes([
            Transform(SchemaNode.product([x, v]), w, p),
            Transform(x, y)
        ])
        self.assertExpectedInline(str(e), """(False, None)""")
