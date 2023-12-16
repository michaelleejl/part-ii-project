import expecttest
import numpy as np
import pandas as pd

from schema import SchemaGraph, SchemaNode, SchemaEdge, AllNodesInClusterMustAlreadyBeInGraphException, \
    AllNodesInFullyConnectedClusterMustHaveSameClusterException, NodeNotInSchemaGraphException, \
    FindingEdgeViaNodeMustRespectEquivalence, Transform, NoShortestPathBetweenNodesException, \
    MultipleShortestPathsBetweenNodesException, CycleDetectedInPathException
from schema.cardinality import Cardinality


class TestSchemaGraph(expecttest.TestCase):

    def test_schemaGraph_addNodeIsIdempotent(self):
        g = SchemaGraph()
        u = SchemaNode("name", cluster="1")
        g.add_node(u)
        g.add_node(u)
        self.assertExpectedInline(str(g.schema_nodes), """[1.name]""")
        self.assertExpectedInline(str(g), """\
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
                                        lambda: g.add_cluster([u, v, w], u),
                                        """When adding a cluster, all nodes in the cluster must already exist in the graph. The following nodes are not in the graph: 1.name3""")

    def test_schemaGraph_addFullyConnectedCluster_succeeds(self):
        g = SchemaGraph()
        u = SchemaNode("name", cluster="1")
        v = SchemaNode("name2", cluster="1")
        g.add_nodes([u, v])
        g.add_cluster([u, v], u)
        self.assertExpectedInline(str(g), """\
ADJACENCY LIST 
==========================

==========================
1.name
--------------------------
name ---> name2
==========================

==========================
1.name2
--------------------------
name ---> name2
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

    def test_findAllEquivalentNodesSuccessfullyFindsAllEquivalentNodesForAtomicNode(self):
        g = SchemaGraph()
        u = SchemaNode("u", cluster="1")
        v = SchemaNode("v", cluster="1")
        g.add_nodes([u, v])
        g.blend_nodes(u, v)
        self.assertExpectedInline(str(list(sorted(g.find_all_equivalent_nodes(u), key=str))), """[1.u, 1.v]""")

    def test_findAllEquivalentNodesSuccessfullyFindsAllEquivalentNodesForProductNode(self):
        g = SchemaGraph()
        u1 = SchemaNode("u1", cluster="1")
        u2 = SchemaNode("u2", cluster="1")
        v1 = SchemaNode("v1", cluster="1")
        v2 = SchemaNode("v2", cluster="1")
        w = SchemaNode("w", cluster="1")
        x = SchemaNode("x", cluster="1")
        y = SchemaNode("y", cluster="1")
        g.add_nodes([u1, u2, v1, v2, w, x, y])
        g.blend_nodes(u1, u2)
        g.blend_nodes(v1, v2)
        g.blend_nodes(x, y)
        p = SchemaNode.product([u1, v2, w, x, y])
        self.assertExpectedInline(str(list(sorted(g.find_all_equivalent_nodes(p), key=str))),
                                  """[1.u1;v1;w;x;x, 1.u1;v1;w;x;y, 1.u1;v1;w;y;x, 1.u1;v1;w;y;y, 1.u1;v2;w;x;x, 1.u1;v2;w;x;y, 1.u1;v2;w;y;x, 1.u1;v2;w;y;y, 1.u2;v1;w;x;x, 1.u2;v1;w;x;y, 1.u2;v1;w;y;x, 1.u2;v1;w;y;y, 1.u2;v2;w;x;x, 1.u2;v2;w;x;y, 1.u2;v2;w;y;x, 1.u2;v2;w;y;y]""")

    def test_findAllShortestPathsBetweenNodes_findsShortestPathOfLengthZero(self):
        g = SchemaGraph()
        u = SchemaNode("u", cluster="1")
        g.add_node(u)
        self.assertExpectedInline(str(g.find_all_shortest_paths_between_nodes(u, u)), """([], [], [], [])""")

    def test_findAllShortestPathsBetweenNodes_findsShortestPathUsingEquivalenceClass(self):
        g = SchemaGraph()
        u = SchemaNode("u", cluster="1")
        v = SchemaNode("v", cluster="1")
        g.add_nodes([u, v])
        g.blend_nodes(u, v)
        self.assertExpectedInline(str(g.find_all_shortest_paths_between_nodes(u, v)), """([1.v], [u === v], [EQU <1.u, 1.v>], [])""")

    def test_findAllShortestPathsBetweenNodes_findsShortestMultiHopPathUsingEquivalenceClass(self):
        g = SchemaGraph()
        u = SchemaNode("u", cluster="1")
        v = SchemaNode("v", cluster="1")
        w = SchemaNode("w", cluster="1")
        g.add_nodes([u, v, w])
        g.blend_nodes(u, v)
        g.add_edge(v, w, Cardinality.ONE_TO_ONE)
        self.assertExpectedInline(str(g.find_all_shortest_paths_between_nodes(u, w)),
                                  """([1.v, 1.w], [u === v, v <--> w], [EQU <1.u, 1.v>, TRV <1.v, 1.w, [], []>], [])""")

    def test_findAllShortestPathsBetweenNodes_FindsProjectionEdgeIfLastEdgeInPath(self):
        g = SchemaGraph()
        u = SchemaNode("u", cluster="1")
        v = SchemaNode("v", cluster="1")
        g.add_nodes([u, v])
        p = SchemaNode.product([u, v])
        self.assertExpectedInline(str(g.find_all_shortest_paths_between_nodes(p, v)), """([1.v], [u;v ---> v], [PRJ <1.u;v, 1.v, [], []>], [])""")

    def test_findAllShortestPathsBetweenNodes_FindsMultiHopPath(self):
        g = SchemaGraph()
        u = SchemaNode("u", cluster="1")
        v1 = SchemaNode("v1", cluster="1")
        v2 = SchemaNode("v2", cluster="1")
        w = SchemaNode("w", cluster="1")
        g.add_nodes([u, v1, v2, w])
        g.blend_nodes(v1, v2)
        p = SchemaNode.product([u, v2])
        g.add_edge(p, w)
        start = SchemaNode.product([u, v1])
        self.assertExpectedInline(str(g.find_all_shortest_paths_between_nodes(start, w)),
                                  """([1.u;v2, 1.w], [u;v1 === u;v2, u;v2 --- w], [EQU <1.u;v1, 1.u;v2>, TRV <1.u;v2, 1.w, [1.w], []>], [1.w])""")

    def test_findAllShortestPathsBetweenNodes_DoesNotFindProjectionEdgeIfProjectionNotLastEdgeInPath(self):
        g = SchemaGraph()
        u = SchemaNode("u", cluster="1")
        v = SchemaNode("v", cluster="1")
        w = SchemaNode("w", cluster="1")
        g.add_nodes([u, v, w])
        p = SchemaNode.product([u, v])
        g.add_edge(v, w)
        self.assertExpectedRaisesInline(
            NoShortestPathBetweenNodesException,
            lambda: str(g.find_all_shortest_paths_between_nodes(p, w)),
            """No paths found between nodes 1.u;v and 1.w.If the path involves a projection that isn't the last edge in the path,The projection will need to be specified as a waypoint.""")

    def test_findAllShortestPathsBetweenNodes_RaisesExceptionIfMultipleShortestPathsFound(self):
        g = SchemaGraph()
        u = SchemaNode("u", cluster="1")
        v = SchemaNode("v", cluster="1")
        w = SchemaNode("w", cluster="2")
        x = SchemaNode("x", cluster="2")
        g.add_nodes([u, v, w, x])
        g.add_edge(u, v)
        g.add_edge(u, w)
        g.add_edge(v, x)
        g.add_edge(w, x)
        self.assertExpectedRaisesInline(
            MultipleShortestPathsBetweenNodesException,
            lambda: str(g.find_all_shortest_paths_between_nodes(u, x)),
            """Multiple shortest paths found between nodes 1.u and 2.x.Please specify one or more waypoints!"""
        )

    def test_findAllShortestPathsBetweenNodes_findsShortestPath_ifMultiplePathsExist(self):
        g = SchemaGraph()
        u = SchemaNode("u", cluster="1")
        v = SchemaNode("v", cluster="1")
        w = SchemaNode("w", cluster="2")
        x = SchemaNode("x", cluster="2")
        y = SchemaNode("y", cluster="3")
        g.add_nodes([u, v, w, x, y])
        g.add_edge(u, v)
        g.add_edge(u, w)
        g.add_edge(v, y, Cardinality.ONE_TO_MANY)
        g.add_edge(w, x, Cardinality.ONE_TO_ONE)
        g.add_edge(x, y)
        self.assertExpectedInline(
            str(g.find_all_shortest_paths_between_nodes(u, y)),
            """([1.v, 3.y], [u --- v, v <--- y], [TRV <1.u, 1.v, [1.v], []>, TRV <1.v, 3.y, [3.y], []>], [1.v, 3.y])"""
        )

    def test_findShortestPath_worksIfNoWaypointsSpecified(self):
        g = SchemaGraph()
        u = SchemaNode("u", cluster="1")
        v = SchemaNode("v", cluster="1")
        w = SchemaNode("w", cluster="2")
        x = SchemaNode("x", cluster="2")
        y = SchemaNode("y", cluster="3")
        g.add_nodes([u, v, w, x, y])
        g.add_edge(u, v)
        g.add_edge(u, w)
        g.add_edge(v, y, Cardinality.ONE_TO_MANY)
        g.add_edge(w, x, Cardinality.ONE_TO_ONE)
        g.add_edge(x, y)
        self.assertExpectedInline(
            str(g.find_shortest_path(u, y, [], False)),
            """([u --- v, v <--- y], [TRV <1.u, 1.v, [1.v], []>, TRV <1.v, 3.y, [3.y], []>], [1.v, 3.y])"""
        )

    def test_findShortestPath_worksIfWaypointsSpecified(self):
        g = SchemaGraph()
        u = SchemaNode("u", cluster="1")
        v = SchemaNode("v", cluster="1")
        w = SchemaNode("w", cluster="1")
        g.add_nodes([u, v, w])
        p = SchemaNode.product([u, v])
        g.add_edge(v, w)
        self.assertExpectedInline(
            str(g.find_shortest_path(p, w, [v], False)),
            """([u;v ---> v, v --- w], [PRJ <1.u;v, 1.v, [], []>, TRV <1.v, 1.w, [1.w], []>], [1.w])""")

    def test_findShortestPath_detectsCycles(self):
        g = SchemaGraph()
        u = SchemaNode("u", cluster="1")
        v = SchemaNode("v", cluster="1")
        g.add_nodes([u, v])
        g.add_edge(u, v)
        self.assertExpectedRaisesInline(
            CycleDetectedInPathException,
            lambda: str(g.find_shortest_path(u, u, [v], False)),
            """Cycle detected in path."""
        )

    def test_add_class_succeeds(self):
        g = SchemaGraph()
        u = SchemaNode("u", cluster="1")
        v = SchemaNode("v", cluster="1")
        schema_class = SchemaNode("class")
        g.add_nodes([u, v])
        g.add_class(schema_class)
        self.assertExpectedInline(str(g), """\
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
Class 1
--------------------------
1.v
==========================

==========================
Class 2
--------------------------
class
==========================

==========================
""")

    def test_findShortestPath_findsMoreDirectPath(self):
        g = SchemaGraph()
        u = SchemaNode("u", cluster="1")
        v = SchemaNode("v", cluster="1")
        w = SchemaNode("w", cluster="1")
        g.add_nodes([u, v, w])
        g.add_edge(u, SchemaNode.product([v, w]))
        g.add_edge(u, w)
        path = g.find_shortest_path(u, w, [], True)
        self.assertExpectedInline(str(path), """([w --- u], [TRV <1.u, 1.w, [1.w], []>], [1.w])""")

    def test_findShortestPath_worksInBackwardsMode(self):
        g = SchemaGraph()
        u = SchemaNode("u", cluster="1")
        v = SchemaNode("v", cluster="1")
        w = SchemaNode("w", cluster="1")
        g.add_nodes([u, v, w])
        g.add_edge(u, w, Cardinality.MANY_TO_ONE)
        path = g.find_shortest_path(u, w, [], True)
        self.assertExpectedInline(str(path), """([w <--- u], [TRV <1.u, 1.w, [1.w], []>], [1.w])""")

    def test_findShortestPath_inBackwardsMode_treatsExpansionAsProjection(self):
        g = SchemaGraph()
        u = SchemaNode("u", cluster="1")
        v = SchemaNode("v", cluster="1")
        w = SchemaNode("w", cluster="1")
        g.add_nodes([u, v, w])
        g.add_edge(u, w, Cardinality.MANY_TO_ONE)
        path = g.find_shortest_path(u, SchemaNode.product([v, w]), [], True)
        self.assertExpectedInline(str(path), """([v;w ---> w, w <--- u], [TRV <1.u, 1.w, [1.w], []>, EXP <1.w, 1.v;w, [], []>], [1.w])""")

    def test_findShortestPath_inBackwardsMode_succeedsWhenAnEquivalenceIsInvolved(self):
        g = SchemaGraph()
        u = SchemaNode("u", cluster="1")
        v = SchemaNode("v", cluster="1")
        w = SchemaNode("w", cluster="1")
        g.add_nodes([u, v, w])
        g.blend_nodes(v, w)
        g.add_edge(u, v, Cardinality.MANY_TO_ONE)
        path = g.find_shortest_path(u, w, [], True)
        self.assertExpectedInline(str(path), """([v <--- u, v === w], [TRV <1.u, 1.v, [1.v], []>, EQU <1.v, 1.w>], [1.v])""")

    def test_get_next_step_forwards_correctly_gets_trv(self):
        g = SchemaGraph()
        u = SchemaNode("u", cluster="1")
        v = SchemaNode("v", cluster="1")
        g.add_nodes([u, v])
        edge = SchemaEdge(u, v, Cardinality.ONE_TO_MANY)
        next_step = g.get_next_step(edge, False)
        self.assertExpectedInline(str(next_step), """TRV <1.u, 1.v, [1.v], []>""")

    def test_get_next_step_forwards_correctly_gets_exp(self):
        g = SchemaGraph()
        u = SchemaNode("u", cluster="1")
        v = SchemaNode("v", cluster="1")
        g.add_nodes([u, v])
        edge = SchemaEdge(u, SchemaNode.product([u, v]), Cardinality.ONE_TO_MANY)
        next_step = g.get_next_step(edge, False)
        self.assertExpectedInline(str(next_step), """EXP <1.u, 1.u;v, [1.v], []>""")

    def test_get_next_step_forwards_correctly_gets_prj(self):
        g = SchemaGraph()
        u = SchemaNode("u", cluster="1")
        v = SchemaNode("v", cluster="1")
        g.add_nodes([u, v])
        edge = SchemaEdge(SchemaNode.product([u, v]), u, Cardinality.MANY_TO_ONE)
        next_step = g.get_next_step(edge, False)
        self.assertExpectedInline(str(next_step), """PRJ <1.u;v, 1.u, [], []>""")

    def test_get_next_step_backwards_correctly_gets_trv(self):
        g = SchemaGraph()
        u = SchemaNode("u", cluster="1")
        v = SchemaNode("v", cluster="1")
        g.add_nodes([u, v])
        edge = SchemaEdge(u, v, Cardinality.ONE_TO_MANY)
        next_step = g.get_next_step(edge, True)
        self.assertExpectedInline(str(next_step), """TRV <1.u, 1.v, [], []>""")

    def test_get_next_step_backwards_correctly_gets_exp(self):
        g = SchemaGraph()
        u = SchemaNode("u", cluster="1")
        v = SchemaNode("v", cluster="1")
        g.add_nodes([u, v])
        edge = SchemaEdge(u, SchemaNode.product([u, v]), Cardinality.ONE_TO_MANY)
        next_step = g.get_next_step(edge, True)
        self.assertExpectedInline(str(next_step), """EXP <1.u, 1.u;v, [], []>""")

    def test_get_next_step_backwards_correctly_gets_prj(self):
        g = SchemaGraph()
        u = SchemaNode("u", cluster="1")
        v = SchemaNode("v", cluster="1")
        g.add_nodes([u, v])
        edge = SchemaEdge(SchemaNode.product([u, v]), u, Cardinality.MANY_TO_ONE)
        next_step = g.get_next_step(edge, True)
        self.assertExpectedInline(str(next_step), """PRJ <1.u;v, 1.u, [1.v], []>""")