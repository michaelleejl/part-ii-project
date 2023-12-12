import expecttest

from schema import SchemaNode, SchemaEdge, Cardinality, SchemaEdgeList, EdgeAlreadyExistsException


class TestSchemaEdgeList(expecttest.TestCase):
    def test_schemaEdgeList_addingToEdgeListSucceedsIfEdgeDoesNotAlreadyExistAndEdgeListEmpty(self):
        u = SchemaNode("name", cluster="1")
        v = SchemaNode("name2", cluster="1")
        e = SchemaEdge(u, v, Cardinality.ONE_TO_MANY)
        es = SchemaEdgeList(frozenset([]))
        self.assertExpectedInline(str(SchemaEdgeList.add_edge(es, e)), """name <--- name2""")

    def test_schemaEdgeList_addingToEdgeListDoesNotMutate(self):
        u = SchemaNode("name", cluster="1")
        v = SchemaNode("name2", cluster="1")
        w = SchemaNode("name3", cluster="1")
        e = SchemaEdge(u, v, Cardinality.ONE_TO_MANY)
        e2 = SchemaEdge(u, w, Cardinality.ONE_TO_ONE)
        es = SchemaEdgeList(frozenset([e]))
        _ = SchemaEdgeList.add_edge(es, e2)
        self.assertExpectedInline(str(es), """name <--- name2""")

    def test_schemaEdgeList_addingToEdgeListRaisesExceptionIfEdgeAlreadyExists(self):
        u = SchemaNode("name", cluster="1")
        v = SchemaNode("name2", cluster="1")
        e = SchemaEdge(u, v, Cardinality.ONE_TO_MANY)
        es = SchemaEdgeList(frozenset([e]))
        e2 = SchemaEdge(u, v, Cardinality.ONE_TO_ONE)
        self.assertExpectedRaisesInline(EdgeAlreadyExistsException, lambda: SchemaEdgeList.add_edge(es, e2), """Edge between 1.name and 1.name2 already exists. Use `replace` instead.""")

    def test_schemaEdgeList_updatingEdgeListSucceeds(self):
        u = SchemaNode("name", cluster="1")
        v = SchemaNode("name2", cluster="1")
        e = SchemaEdge(u, v, Cardinality.ONE_TO_MANY)
        es = SchemaEdgeList(frozenset([e]))
        e2 = SchemaEdge(u, v, Cardinality.ONE_TO_ONE)
        self.assertExpectedInline(str(SchemaEdgeList.replace_edge(es, e2)), """name <--> name2""")