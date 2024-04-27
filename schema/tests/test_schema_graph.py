import expecttest


from schema.cardinality import Cardinality
from schema.exceptions import (
    AllNodesInClusterMustAlreadyBeInGraphException,
    NodeNotInSchemaGraphException,
    MultipleShortestPathsBetweenNodesException,
    NoShortestPathBetweenNodesException,
    CycleDetectedInPathException,
)
from schema.graph import SchemaGraph
from schema.node import AtomicNode, SchemaNode, SchemaClass


class TestSchemaGraph(expecttest.TestCase):

    def test_schemaGraph_addNodeIsIdempotent(self):
        g = SchemaGraph()
        u = AtomicNode("name")
        g.add_node(u)
        g.add_node(u)
        u.id_prefix = 0
        self.assertExpectedInline(str([n.name for n in g.schema_nodes]), """['name']""")
        self.assertExpectedInline(
            str(g),
            """\
ADJACENCY LIST 
==========================


==========================

EQUIVALENCE CLASSES 
==========================

==========================
Class 0
--------------------------
name
==========================

==========================
""",
        )

    def test_schemaGraph_addNodesActsLikeSetUnion(self):
        g = SchemaGraph()
        u = AtomicNode("name")
        v = AtomicNode("name2")
        w = AtomicNode("name3")
        u.id_prefix = 0
        v.id_prefix = 0
        w.id_prefix = 0
        g.add_nodes([u, v])
        g.add_nodes([v, w])
        self.assertExpectedInline(str(g.schema_nodes), """[name, name2, name3]""")
        self.assertExpectedInline(
            str(g),
            """\
ADJACENCY LIST 
==========================


==========================

EQUIVALENCE CLASSES 
==========================

==========================
Class 0
--------------------------
name
==========================

==========================
Class 1
--------------------------
name2
==========================

==========================
Class 2
--------------------------
name3
==========================

==========================
""",
        )

    def test_schemaGraph_blendNodesMergesEquivalenceClasses(self):
        g = SchemaGraph()
        u = AtomicNode("name")
        v = AtomicNode("name2")
        u.id_prefix = 0
        v.id_prefix = 0
        g.add_nodes([u, v])
        g.blend_nodes(u, v)
        self.assertExpectedInline(
            str(g),
            """\
ADJACENCY LIST 
==========================


==========================

EQUIVALENCE CLASSES 
==========================

==========================
Class 0
--------------------------
name
name2
==========================

==========================
""",
        )

    def test_schemaGraph_areNodesEqual_returnsTrue_ifNodesInSameEquivalenceClass(self):
        g = SchemaGraph()
        u = AtomicNode("name")
        v = AtomicNode("name2")
        g.add_nodes([u, v])
        g.blend_nodes(u, v)
        self.assertTrue(g.are_nodes_equal(u, v))

    def test_schemaGraph_areNodesEqual_returnsFalse_ifNodesInDifferentEquivalenceClasses(
        self,
    ):
        g = SchemaGraph()
        u = AtomicNode("name")
        v = AtomicNode("name2")
        g.add_nodes([u, v])
        self.assertFalse(g.are_nodes_equal(u, v))

    def test_schemaGraph_addFullyConnectedCluster_raisesExceptionIfNodesNotInGraph(
        self,
    ):
        g = SchemaGraph()
        u = AtomicNode("name")
        v = AtomicNode("name2")
        w = AtomicNode("name3")
        u.id_prefix = 0
        v.id_prefix = 0
        w.id_prefix = 0
        g.add_nodes([u, v])
        self.assertExpectedRaisesInline(
            AllNodesInClusterMustAlreadyBeInGraphException,
            lambda: g.add_cluster([u, v, w], u),
            """When adding a cluster, all nodes in the cluster must already exist in the graph. The following nodes are not in the graph: name3""",
        )

    def test_schemaGraph_addFullyConnectedCluster_succeeds(self):
        g = SchemaGraph()
        u = AtomicNode("name")
        v = AtomicNode("name2")
        u.id_prefix = 0
        v.id_prefix = 0
        g.add_nodes([u, v])
        g.add_cluster([u, v], u)
        self.assertExpectedInline(
            str(g),
            """\
ADJACENCY LIST 
==========================

==========================
name
--------------------------
name ---> name2
==========================

==========================
name2
--------------------------
name2 <--- name
==========================

==========================

EQUIVALENCE CLASSES 
==========================

==========================
Class 0
--------------------------
name
==========================

==========================
Class 1
--------------------------
name2
==========================

==========================
""",
        )

    def test_schemaGraph_addEdgeRaisesException_ifFromNodeNotInGraph(self):
        g = SchemaGraph()
        u = AtomicNode("u")
        v = AtomicNode("v")
        u.id = "000"
        self.assertExpectedRaisesInline(
            NodeNotInSchemaGraphException,
            lambda: g.add_edge(u, v),
            """Node u <000> is not in schema graph.""",
        )

    def test_schemaGraph_addEdgeRaisesException_ifToNodeNotInGraph(self):
        g = SchemaGraph()
        u = AtomicNode("u")
        v = AtomicNode("v")
        v.id = "001"
        g.add_node(u)
        self.assertExpectedRaisesInline(
            NodeNotInSchemaGraphException,
            lambda: g.add_edge(u, v),
            """Node v <001> is not in schema graph.""",
        )

    def test_schemaGraph_addEdgeDoesNothing_ifFromNodeEqualsToNode(self):
        g = SchemaGraph()
        u = AtomicNode("u")
        u.id_prefix = 0
        g.add_node(u)
        g.add_edge(u, u)
        self.assertExpectedInline(
            str(g),
            """\
ADJACENCY LIST 
==========================


==========================

EQUIVALENCE CLASSES 
==========================

==========================
Class 0
--------------------------
u
==========================

==========================
""",
        )

    def test_schemaGraphAddEdge_successfullyAddsNewEdgeToAdjacencyList(self):
        g = SchemaGraph()
        u = AtomicNode("u")
        v = AtomicNode("v")
        u.id_prefix = 0
        v.id_prefix = 0
        g.add_node(u)
        g.add_node(v)
        g.add_edge(u, v, Cardinality.ONE_TO_ONE)
        self.assertExpectedInline(
            str(g),
            """\
ADJACENCY LIST 
==========================

==========================
u
--------------------------
u <--> v
==========================

==========================
v
--------------------------
v <--> u
==========================

==========================

EQUIVALENCE CLASSES 
==========================

==========================
Class 0
--------------------------
u
==========================

==========================
Class 1
--------------------------
v
==========================

==========================
""",
        )

    def test_findAllEquivalentNodesSuccessfullyFindsAllEquivalentNodesForAtomicNode(
        self,
    ):
        g = SchemaGraph()
        u = AtomicNode("u")
        v = AtomicNode("v")
        u.id = "000"
        v.id = "001"
        g.add_nodes([u, v])
        g.blend_nodes(u, v)
        self.assertExpectedInline(
            str(list(sorted(g.find_all_equivalent_nodes(u), key=str))),
            """[u <000>, v <001>]""",
        )

    def test_findAllEquivalentNodesSuccessfullyFindsAllEquivalentNodesForProductNode(
        self,
    ):
        g = SchemaGraph()
        u1 = AtomicNode("u1")
        u1.id_prefix = 0
        u2 = AtomicNode("u2")
        u2.id_prefix = 0
        v1 = AtomicNode("v1")
        v1.id_prefix = 0
        v2 = AtomicNode("v2")
        v2.id_prefix = 0
        w = AtomicNode("w")
        w.id_prefix = 0
        x = AtomicNode("x")
        x.id_prefix = 0
        y = AtomicNode("y")
        y.id_prefix = 0
        g.add_nodes([u1, u2, v1, v2, w, x, y])
        g.blend_nodes(u1, u2)
        g.blend_nodes(v1, v2)
        g.blend_nodes(x, y)
        p = SchemaNode.product([u1, v2, w, x, y])
        self.assertExpectedInline(
            str(list(sorted(g.find_all_equivalent_nodes(p), key=str))),
            """[u1;v1;w;x;x, u1;v1;w;x;y, u1;v1;w;y;x, u1;v1;w;y;y, u1;v2;w;x;x, u1;v2;w;x;y, u1;v2;w;y;x, u1;v2;w;y;y, u2;v1;w;x;x, u2;v1;w;x;y, u2;v1;w;y;x, u2;v1;w;y;y, u2;v2;w;x;x, u2;v2;w;x;y, u2;v2;w;y;x, u2;v2;w;y;y]""",
        )

    def test_findAllShortestPathsBetweenNodes_findsShortestPathOfLengthZero(self):
        g = SchemaGraph()
        u = AtomicNode("u")
        g.add_node(u)
        self.assertExpectedInline(
            str(g.find_all_shortest_paths_between_nodes(u, u)), """([], [])"""
        )

    def test_findAllShortestPathsBetweenNodes_findsShortestPathUsingEquivalenceClass(
        self,
    ):
        g = SchemaGraph()
        u = AtomicNode("u")
        u.id = "000"
        v = AtomicNode("v")
        v.id = "001"
        g.add_nodes([u, v])
        g.blend_nodes(u, v)
        self.assertExpectedInline(
            str(g.find_all_shortest_paths_between_nodes(u, v)),
            """([v <001>], [u <000> === v <001>])""",
        )

    def test_findAllShortestPathsBetweenNodes_findsShortestMultiHopPathUsingEquivalenceClass(
        self,
    ):
        g = SchemaGraph()
        u = AtomicNode("u")
        u.id = "000"
        v = AtomicNode("v")
        v.id = "001"
        w = AtomicNode("w")
        w.id = "002"
        g.add_nodes([u, v, w])
        g.blend_nodes(u, v)
        g.add_edge(v, w, Cardinality.ONE_TO_ONE)
        self.assertExpectedInline(
            str(g.find_all_shortest_paths_between_nodes(u, w)),
            """([v <001>, w <002>], [u <000> === v <001>, v <001> <--> w <002>])""",
        )

    def test_findAllShortestPathsBetweenNodes_FindsMultiHopPath(self):
        g = SchemaGraph()
        u = AtomicNode("u")
        u.id = "000"
        v1 = AtomicNode("v1")
        v1.id = "001"
        v2 = AtomicNode("v2")
        v2.id = "002"
        w = AtomicNode("w")
        w.id = "003"
        g.add_nodes([u, v1, v2, w])
        g.blend_nodes(v1, v2)
        p = SchemaNode.product([u, v2])
        g.add_edge(p, w)
        start = SchemaNode.product([u, v1])
        self.assertExpectedInline(
            str(g.find_all_shortest_paths_between_nodes(start, w)),
            """([u <000>;v2 <002>, w <003>], [u <000>;v1 <001> === u <000>;v2 <002>, u <000>;v2 <002> --- w <003>])""",
        )

    def test_findAllShortestPathsBetweenNodes_DoesNotFindProjectionEdgeIfProjectionNotLastEdgeInPath(
        self,
    ):
        g = SchemaGraph()
        u = AtomicNode("u")
        u.id_prefix = 0
        v = AtomicNode("v")
        v.id_prefix = 0
        w = AtomicNode("w")
        w.id_prefix = 0
        g.add_nodes([u, v, w])
        p = SchemaNode.product([u, v])
        g.add_edge(v, w)
        self.assertExpectedRaisesInline(
            NoShortestPathBetweenNodesException,
            lambda: str(g.find_all_shortest_paths_between_nodes(p, w)),
            """No paths found between nodes u;v and w. If the path involves a projection, the projection will need to be specified as a waypoint.""",
        )

    def test_findAllShortestPathsBetweenNodes_RaisesExceptionIfMultipleShortestPathsFound(
        self,
    ):
        g = SchemaGraph()
        u = AtomicNode("u")
        v = AtomicNode("v")
        w = AtomicNode("w")
        x = AtomicNode("x")
        u.id_prefix = 0
        v.id_prefix = 0
        w.id_prefix = 0
        x.id_prefix = 0
        g.add_nodes([u, v, w, x])
        g.add_edge(u, v)
        g.add_edge(u, w)
        g.add_edge(v, x)
        g.add_edge(w, x)
        self.assertExpectedRaisesInline(
            MultipleShortestPathsBetweenNodesException,
            lambda: str(g.find_all_shortest_paths_between_nodes(u, x)),
            """Multiple shortest paths found between nodes u and x. Shortest paths: [[u --- v, v --- x], [u --- w, w --- x]]""",
        )

    def test_findAllShortestPathsBetweenNodes_findsShortestPath_ifMultiplePathsExist(
        self,
    ):
        g = SchemaGraph()
        u = AtomicNode("u")
        v = AtomicNode("v")
        w = AtomicNode("w")
        x = AtomicNode("x")
        y = AtomicNode("y")
        u.id_prefix = 0
        v.id_prefix = 0
        w.id_prefix = 0
        x.id_prefix = 0
        y.id_prefix = 0
        g.add_nodes([u, v, w, x, y])
        g.add_edge(u, v)
        g.add_edge(u, w)
        g.add_edge(v, y, Cardinality.ONE_TO_MANY)
        g.add_edge(w, x, Cardinality.ONE_TO_ONE)
        g.add_edge(x, y)
        self.assertExpectedInline(
            str(g.find_all_shortest_paths_between_nodes(u, y)),
            """([v, y], [u --- v, v <--- y])""",
        )

    def test_findShortestPath_worksIfNoWaypointsSpecified(self):
        g = SchemaGraph()
        u = AtomicNode("u")
        v = AtomicNode("v")
        w = AtomicNode("w")
        x = AtomicNode("x")
        y = AtomicNode("y")
        u.id_prefix = 0
        v.id_prefix = 0
        w.id_prefix = 0
        x.id_prefix = 0
        y.id_prefix = 0
        g.add_nodes([u, v, w, x, y])
        g.add_edge(u, v)
        g.add_edge(u, w)
        g.add_edge(v, y, Cardinality.ONE_TO_MANY)
        g.add_edge(w, x, Cardinality.ONE_TO_ONE)
        g.add_edge(x, y)
        self.assertExpectedInline(
            str(g.find_shortest_path(u, y, [])),
            """(<Cardinality.MANY_TO_MANY: 4>, [u --- v, v <--- y])""",
        )

    def test_findShortestPath_worksIfWaypointsSpecified(self):
        g = SchemaGraph()
        u = AtomicNode("u")
        v = AtomicNode("v")
        w = AtomicNode("w")
        u.id_prefix = 0
        v.id_prefix = 0
        w.id_prefix = 0
        g.add_nodes([u, v, w])
        g.add_edge(u, v)
        g.add_edge(v, w)
        self.assertExpectedInline(
            str(g.find_shortest_path(u, w, [v])),
            """(<Cardinality.MANY_TO_MANY: 4>, [u --- v, v --- w])""",
        )

    def test_findShortestPath_detectsCycles(self):
        g = SchemaGraph()
        u = AtomicNode("u")
        v = AtomicNode("v")
        u.id_prefix = 0
        v.id_prefix = 0
        g.add_nodes([u, v])
        g.add_edge(u, v)
        self.assertExpectedRaisesInline(
            CycleDetectedInPathException,
            lambda: str(g.find_shortest_path(u, u, [v])),
            """Cycle detected in path.""",
        )

    def test_add_class_succeeds(self):
        g = SchemaGraph()
        u = AtomicNode("u")
        v = AtomicNode("v")
        u.id_prefix = 0
        v.id_prefix = 0
        schema_class = SchemaClass("class")
        g.add_nodes([u, v])
        g.add_class(schema_class)
        self.assertExpectedInline(
            str(g),
            """\
ADJACENCY LIST 
==========================


==========================

EQUIVALENCE CLASSES 
==========================

==========================
Class 0
--------------------------
u
==========================

==========================
Class 1
--------------------------
v
==========================

==========================
Class 2
--------------------------
class
==========================

==========================
""",
        )

    def test_findShortestPath_findsMoreDirectPath(self):
        g = SchemaGraph()
        u = AtomicNode("u")
        v = AtomicNode("v")
        w = AtomicNode("w")
        u.id_prefix = 0
        v.id_prefix = 0
        w.id_prefix = 0
        g.add_nodes([u, v, w])
        g.add_edge(u, SchemaNode.product([v, w]))
        g.add_edge(u, w)
        path = g.find_shortest_path(u, w, [])
        self.assertExpectedInline(
            str(path), """(<Cardinality.MANY_TO_MANY: 4>, [u --- w])"""
        )
