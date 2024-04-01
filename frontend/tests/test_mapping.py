import expecttest

from representation.domain import Domain
from representation.mapping import Mapping
from schema.cardinality import Cardinality
from schema.edge import SchemaEdge
from schema.node import AtomicNode, SchemaNode


class TestMapping(expecttest.TestCase):
    def test_init(self):
        u = AtomicNode("u")
        v = AtomicNode("v")

        u.id_prefix = 0
        v.id_prefix = 0

        edge = SchemaEdge(u, v, cardinality=Cardinality.ONE_TO_ONE)
        mapping = Mapping(edge)

        self.assertExpectedInline(str(mapping), """u <--> v""")

    def test_curry_withHiddenKeys(self):
        u = AtomicNode("u")
        v = AtomicNode("v")
        w = AtomicNode("w")

        u.id_prefix = 0
        v.id_prefix = 0
        w.id_prefix = 0

        uv = SchemaNode.product([u, v])

        edge = SchemaEdge(uv, w, cardinality=Cardinality.ONE_TO_MANY)
        mapping = Mapping(edge, hidden_keys=[Domain("c", w)])

        b = Domain("b", v)

        new_mapping = mapping.curry(1, b)
        self.assertExpectedInline(str(new_mapping), """u --- w""")
        self.assertExpectedInline(str(new_mapping.hidden_keys), """[b, c]""")
        self.assertExpectedInline(str(new_mapping.transform), """[CUR <1, b>]""")

    def test_uncurry_withSingleHiddenKey(self):
        u = AtomicNode("u")
        v = AtomicNode("v")

        u.id_prefix = 0
        v.id_prefix = 0

        b = Domain("b", v)

        edge = SchemaEdge(u, v, cardinality=Cardinality.ONE_TO_MANY)
        mapping = Mapping(edge, hidden_keys=[b])

        new_mapping = mapping.uncurry(b)
        self.assertExpectedInline(str(new_mapping), """u;v ---> v""")
        self.assertExpectedInline(str(new_mapping.hidden_keys), """[]""")
        self.assertExpectedInline(str(new_mapping.transform), """[UNC <0, 1>]""")

    def test_uncurry_withMultipleHiddenKeys(self):
        u = AtomicNode("u")
        v = AtomicNode("v")
        w = AtomicNode("w")
        vw = SchemaNode.product([v, w])

        u.id_prefix = 0
        v.id_prefix = 0
        w.id_prefix = 0

        b = Domain("b", v)
        c = Domain("c", w)

        edge = SchemaEdge(u, vw, cardinality=Cardinality.ONE_TO_MANY)
        mapping = Mapping(edge, hidden_keys=[b, c])

        new_mapping = mapping.uncurry(c)
        self.assertExpectedInline(str(new_mapping), """u;w <--- v;w""")
        self.assertExpectedInline(str(new_mapping.hidden_keys), """[b]""")
        self.assertExpectedInline(str(new_mapping.transform), """[UNC <1, 1>]""")

    def test_curry_withNoHiddenKeys(self):
        u = AtomicNode("u")
        v = AtomicNode("v")
        w = AtomicNode("w")

        u.id_prefix = 0
        v.id_prefix = 0
        w.id_prefix = 0

        uv = SchemaNode.product([u, v])

        edge = SchemaEdge(uv, w, cardinality=Cardinality.MANY_TO_ONE)
        mapping = Mapping(edge)

        b = Domain("b", v)

        new_mapping = mapping.curry(1, b)
        self.assertExpectedInline(str(new_mapping), """u --- w""")
        self.assertExpectedInline(str(new_mapping.hidden_keys), """[b]""")
        self.assertExpectedInline(str(new_mapping.transform), """[CUR <1, b>]""")

    def test_carry_simple(self):
        u = AtomicNode("u")
        v = AtomicNode("v")
        w = AtomicNode("w")

        u.id_prefix = 0
        v.id_prefix = 0
        w.id_prefix = 0

        edge = SchemaEdge(u, v, cardinality=Cardinality.MANY_TO_ONE)
        mapping = Mapping(edge)

        c = Domain("c", w)

        new_mapping = mapping.carry(c)
        self.assertExpectedInline(str(new_mapping), """u;w ---> v;w""")
        self.assertExpectedInline(str(new_mapping.hidden_keys), """[]""")
        self.assertExpectedInline(str(new_mapping.transform), """[CAR <c, 1, 1>]""")

    def test_carry_withUnequalNumberOfStartAndEndNodes(self):
        u = AtomicNode("u")
        v = AtomicNode("v")
        w = AtomicNode("w")
        x = AtomicNode("x")

        u.id_prefix = 0
        v.id_prefix = 0
        w.id_prefix = 0
        x.id_prefix = 0

        uv = SchemaNode.product([u, v])

        edge = SchemaEdge(uv, w, cardinality=Cardinality.ONE_TO_MANY)
        mapping = Mapping(edge)

        c = Domain("c", x)

        new_mapping = mapping.carry(c)
        self.assertExpectedInline(str(new_mapping), """u;v;x <--- w;x""")
        self.assertExpectedInline(str(new_mapping.hidden_keys), """[]""")
        self.assertExpectedInline(str(new_mapping.transform), """[CAR <c, 2, 1>]""")
        self.assertExpectedInline(str(new_mapping.carried), """{c: (2, 1)}""")

    def test_carry_twiceSucceeds(self):
        u = AtomicNode("u")
        v = AtomicNode("v")
        w = AtomicNode("w")
        x = AtomicNode("x")

        u.id_prefix = 0
        v.id_prefix = 0
        w.id_prefix = 0
        x.id_prefix = 0

        edge = SchemaEdge(u, v, cardinality=Cardinality.MANY_TO_ONE)
        mapping = Mapping(edge)

        c = Domain("c", w)
        d = Domain("d", w)

        new_mapping = mapping.carry(c).carry(d)
        self.assertExpectedInline(str(new_mapping), """u;w;w ---> v;w;w""")
        self.assertExpectedInline(str(new_mapping.hidden_keys), """[]""")
        self.assertExpectedInline(
            str(new_mapping.transform), """[CAR <c, 1, 1>, CAR <d, 2, 2>]"""
        )
        self.assertExpectedInline(
            str(new_mapping.carried), """{c: (1, 1), d: (2, 2)}"""
        )

    def test_carry_raisesAssertionError_ifAlreadyCarried(self):
        u = AtomicNode("u")
        v = AtomicNode("v")
        w = AtomicNode("w")

        u.id_prefix = 0
        v.id_prefix = 0
        w.id_prefix = 0

        edge = SchemaEdge(u, v, cardinality=Cardinality.MANY_TO_ONE)
        mapping = Mapping(edge, carried={Domain("c", w): (1, 1)})

        c = Domain("c", w)

        with self.assertRaises(AssertionError):
            mapping.carry(c)

    def test_drop_simple(self):
        u = AtomicNode("u")
        v = AtomicNode("v")
        w = AtomicNode("w")

        u.id_prefix = 0
        v.id_prefix = 0
        w.id_prefix = 0

        edge = SchemaEdge(u, v, cardinality=Cardinality.MANY_TO_ONE)
        mapping = Mapping(edge)

        c = Domain("c", w)

        new_mapping = mapping.carry(c)

        new_mapping = new_mapping.drop(c)

        self.assertExpectedInline(str(new_mapping), """u ---> v""")
        self.assertExpectedInline(str(new_mapping.hidden_keys), """[]""")
        self.assertExpectedInline(
            str(new_mapping.transform), """[CAR <c, 1, 1>, DRP <1, 1>]"""
        )
        self.assertExpectedInline(str(new_mapping.carried), """{}""")

    def test_drop_complex(self):
        u = AtomicNode("u")
        v = AtomicNode("v")
        w = AtomicNode("w")
        x = AtomicNode("x")

        u.id_prefix = 0
        v.id_prefix = 0
        w.id_prefix = 0
        x.id_prefix = 0

        edge = SchemaEdge(u, v, cardinality=Cardinality.MANY_TO_ONE)
        mapping = Mapping(edge)

        c = Domain("c", w)
        d = Domain("d", w)

        new_mapping = mapping.carry(c).carry(d).drop(c)

        self.assertExpectedInline(str(new_mapping), """u;w ---> v;w""")
        self.assertExpectedInline(str(new_mapping.hidden_keys), """[]""")
        self.assertExpectedInline(
            str(new_mapping.transform), """[CAR <c, 1, 1>, CAR <d, 2, 2>, DRP <1, 1>]"""
        )
        self.assertExpectedInline(str(new_mapping.carried), """{d: (1, 1)}""")

    def test_carry_then_curry(self):
        u = AtomicNode("u")
        v = AtomicNode("v")
        w = AtomicNode("w")
        x = AtomicNode("x")

        u.id_prefix = 0
        v.id_prefix = 0
        w.id_prefix = 0
        x.id_prefix = 0

        edge = SchemaEdge(u, v, cardinality=Cardinality.MANY_TO_ONE)
        mapping = Mapping(edge)

        c = Domain("c", w)
        d = Domain("d", w)

        new_mapping = mapping.carry(c).carry(d).curry(1, c)
        self.assertExpectedInline(str(new_mapping), """u;w --- v;w;w""")
        self.assertExpectedInline(str(new_mapping.hidden_keys), """[c]""")
        self.assertExpectedInline(
            str(new_mapping.transform), """[CAR <c, 1, 1>, CAR <d, 2, 2>, CUR <1, c>]"""
        )
        self.assertExpectedInline(
            str(new_mapping.carried), """{c: (-1, 1), d: (1, 2)}"""
        )

    def test_carry_then_curry_then_curry(self):
        u = AtomicNode("u")
        v = AtomicNode("v")
        w = AtomicNode("w")
        x = AtomicNode("x")

        u.id_prefix = 0
        v.id_prefix = 0
        w.id_prefix = 0
        x.id_prefix = 0

        edge = SchemaEdge(u, v, cardinality=Cardinality.MANY_TO_ONE)
        mapping = Mapping(edge)

        a = Domain("a", u)
        c = Domain("c", w)
        d = Domain("d", w)
        new_mapping = mapping.carry(c).carry(d).curry(1, c).curry(0, a)

        self.assertExpectedInline(str(new_mapping), """w --- v;w;w""")
        self.assertExpectedInline(str(new_mapping.hidden_keys), """[a, c]""")
        self.assertExpectedInline(
            str(new_mapping.transform),
            """[CAR <c, 1, 1>, CAR <d, 2, 2>, CUR <1, c>, CUR <0, a>]""",
        )
        self.assertExpectedInline(
            str(new_mapping.carried), """{c: (-2, 1), d: (0, 2)}"""
        )

    def test_carry_then_curry_then_curry_then_uncurry(self):
        u = AtomicNode("u")
        v = AtomicNode("v")
        w = AtomicNode("w")
        x = AtomicNode("x")

        u.id_prefix = 0
        v.id_prefix = 0
        w.id_prefix = 0
        x.id_prefix = 0

        edge = SchemaEdge(u, v, cardinality=Cardinality.MANY_TO_ONE)
        mapping = Mapping(edge)

        a = Domain("a", u)
        c = Domain("c", w)
        d = Domain("d", w)
        new_mapping = mapping.carry(c).carry(d).curry(1, c).curry(0, a).uncurry(a)

        self.assertExpectedInline(str(new_mapping), """w;u --- v;w;w""")
        self.assertExpectedInline(str(new_mapping.hidden_keys), """[c]""")
        self.assertExpectedInline(
            str(new_mapping.transform),
            """[CAR <c, 1, 1>, CAR <d, 2, 2>, CUR <1, c>, CUR <0, a>, UNC <0, 1>]""",
        )
        self.assertExpectedInline(
            str(new_mapping.carried), """{c: (-1, 1), d: (0, 2)}"""
        )

    def test_invert_simple(self):
        u = AtomicNode("u")
        v = AtomicNode("v")

        u.id_prefix = 0
        v.id_prefix = 0

        edge = SchemaEdge(u, v, cardinality=Cardinality.MANY_TO_ONE)
        mapping = Mapping(edge)

        new_mapping, namespace = mapping.invert()(set(), lambda x, y: y)

        self.assertExpectedInline(str(new_mapping), """v <--- u""")
        self.assertExpectedInline(str(new_mapping.hidden_keys), """[u]""")
        self.assertExpectedInline(
            str(new_mapping.transform), """[INV <[u], 1, 1, []>]"""
        )

    def test_invert_oneToOne(self):
        u = AtomicNode("u")
        v = AtomicNode("v")

        u.id_prefix = 0
        v.id_prefix = 0

        edge = SchemaEdge(u, v, cardinality=Cardinality.ONE_TO_ONE)
        mapping = Mapping(edge)

        new_mapping, namespace = mapping.invert()(set(), lambda x, y: y)

        self.assertExpectedInline(str(new_mapping), """v <--> u""")
        self.assertExpectedInline(str(new_mapping.hidden_keys), """[]""")
        self.assertExpectedInline(
            str(new_mapping.transform), """[INV <[], 1, 1, []>]"""
        )

    def test_invert_excludesCarried(self):
        u = AtomicNode("u")
        v = AtomicNode("v")
        w = AtomicNode("w")
        x = AtomicNode("x")

        u.id_prefix = 0
        v.id_prefix = 0
        w.id_prefix = 0
        x.id_prefix = 0

        uv = SchemaNode.product([u, v])

        edge = SchemaEdge(uv, w, cardinality=Cardinality.MANY_TO_ONE)
        mapping = Mapping(edge)

        new_mapping = mapping.carry(Domain("c", x))

        new_mapping, namespace = new_mapping.invert()(set(), lambda x, y: y)

        self.assertExpectedInline(str(new_mapping), """w;x <--- u;v;x""")
        self.assertExpectedInline(str(new_mapping.hidden_keys), """[u, v]""")
        self.assertExpectedInline(
            str(new_mapping.transform), """[CAR <c, 2, 1>, INV <[u, v], 3, 2, [2]>]"""
        )

    def test_invert_doesNotExcludeCarriedIfCarriedIsCurried(self):
        u = AtomicNode("u")
        v = AtomicNode("v")
        w = AtomicNode("w")
        x = AtomicNode("x")

        u.id_prefix = 0
        v.id_prefix = 0
        w.id_prefix = 0
        x.id_prefix = 0

        uv = SchemaNode.product([u, v])

        edge = SchemaEdge(uv, w, cardinality=Cardinality.MANY_TO_ONE)
        mapping = Mapping(edge)

        new_mapping = mapping.carry(Domain("c", x)).curry(2, Domain("c", x))

        new_mapping, namespace = new_mapping.invert()(set(), lambda x, y: y)

        self.assertExpectedInline(str(new_mapping), """w;x --- u;v""")
        self.assertExpectedInline(str(new_mapping.hidden_keys), """[u, v]""")
        self.assertExpectedInline(
            str(new_mapping.transform),
            """[CAR <c, 2, 1>, CUR <2, c>, INV <[u, v], 2, 2, []>]""",
        )
