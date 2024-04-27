import expecttest

from schema.cardinality import Cardinality
from schema.edge import SchemaEdge
from schema.edge_list import SchemaEdgeList
from schema.exceptions import EdgeAlreadyExistsException
from schema.node import AtomicNode


class TestSchemaEdgeList(expecttest.TestCase):
    def test_schemaEdgeList_addingToEdgeListSucceedsIfEdgeDoesNotAlreadyExistAndEdgeListEmpty(
        self,
    ):
        u = AtomicNode("name")
        v = AtomicNode("name2")
        u.id = "000"
        v.id = "001"
        e = SchemaEdge(u, v, Cardinality.ONE_TO_MANY)
        es = SchemaEdgeList(frozenset([]))
        self.assertExpectedInline(
            str(SchemaEdgeList.add_edge(es, e)), """name <000> <--- name2 <001>"""
        )

    def test_schemaEdgeList_addingToEdgeListDoesNotMutate(self):
        u = AtomicNode("name")
        v = AtomicNode("name2")
        u.id = "000"
        v.id = "001"
        w = AtomicNode("name3")
        e = SchemaEdge(u, v, Cardinality.ONE_TO_MANY)
        e2 = SchemaEdge(u, w, Cardinality.ONE_TO_ONE)
        es = SchemaEdgeList(frozenset([e]))
        _ = SchemaEdgeList.add_edge(es, e2)
        self.assertExpectedInline(str(es), """name <000> <--- name2 <001>""")

    def test_schemaEdgeList_addingToEdgeListRaisesExceptionIfEdgeAlreadyExists(self):
        u = AtomicNode("name")
        v = AtomicNode("name2")
        u.id = "000"
        v.id = "001"
        e = SchemaEdge(u, v, Cardinality.ONE_TO_MANY)
        es = SchemaEdgeList(frozenset([e]))
        e2 = SchemaEdge(u, v, Cardinality.ONE_TO_ONE)
        self.assertExpectedRaisesInline(
            EdgeAlreadyExistsException,
            lambda: SchemaEdgeList.add_edge(es, e2),
            """Edge between name <000> and name2 <001> already exists. Use `replace` instead.""",
        )

    def test_schemaEdgeList_updatingEdgeListSucceeds(self):
        u = AtomicNode("name")
        v = AtomicNode("name2")
        u.id = "000"
        v.id = "001"
        e = SchemaEdge(u, v, Cardinality.ONE_TO_MANY)
        es = SchemaEdgeList(frozenset([e]))
        e2 = SchemaEdge(u, v, Cardinality.ONE_TO_ONE)
        self.assertExpectedInline(
            str(SchemaEdgeList.replace_edge(es, e2)), """name <000> <--> name2 <001>"""
        )
