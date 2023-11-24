import unittest

import pandas as pd

from schema import SchemaNode
from union_find.union_find import UnionFindItem, UnionFind


class UnionFindTest(unittest.TestCase):
    def test_item_equality_isValueEquality_whenNodeEqual(self):
        u = SchemaNode("name", pd.DataFrame([0, 1, 2]), family="1")
        i1 = UnionFindItem(2, u)
        i2 = UnionFindItem(2, u)
        i3 = UnionFindItem(3, u)
        self.assertEqual(i1, i2)
        self.assertNotEqual(i1, i3)

    def test_item_equality_isNotValueEquality_whenNodeUnequal(self):
        u = SchemaNode("name", pd.DataFrame([0, 1, 2]), family="1")
        v = SchemaNode("name2", pd.DataFrame([0, 1, 2]), family="1")
        i1 = UnionFindItem(2, u)
        i2 = UnionFindItem(2, v)
        self.assertNotEqual(i1, i2)

    def test_add_singleton_succeedsWithNoSideEffects_whenSingletonNotInUF(self):
        uf = UnionFind.initialise()
        u = SchemaNode("name", pd.DataFrame([0, 1, 2]), family="1")
        uf2 = UnionFind.add_singleton(uf, 2, u)
        self.assertEqual({UnionFindItem(2, u): UnionFindItem(2, u)}, uf2.leaders)
        self.assertEqual({UnionFindItem(2, u): 0}, uf2.rank)
        self.assertEqual({UnionFindItem(2, u): frozenset()}, uf2.graph)
        self.assertEqual({}, uf.leaders)

    def test_add_singleton_isIdempotent(self):
        uf = UnionFind.initialise()
        u = SchemaNode("name", pd.DataFrame([0, 1, 2]), family="1")
        uf2 = UnionFind.add_singleton(uf, 2, u)
        uf3 = UnionFind.add_singleton(uf2, 2, u)
        self.assertEqual({UnionFindItem(2, u): UnionFindItem(2, u)}, uf3.leaders)
        self.assertEqual({UnionFindItem(2, u): 0}, uf3.rank)
        self.assertEqual({UnionFindItem(2, u): frozenset()}, uf3.graph)
        self.assertEqual({}, uf.leaders)

    def test_union_succeeds(self):
        uf = UnionFind.initialise()
        u = SchemaNode("name", pd.DataFrame([0, 1, 2]), family="1")
        uf2 = UnionFind.add_singleton(uf, 2, u)
        uf3 = UnionFind.add_singleton(uf2, 3, u)
        uf4 = UnionFind.union(uf3, UnionFindItem(2, u), UnionFindItem(3, u))
        self.assertEqual({UnionFindItem(2, u): UnionFindItem(2, u), UnionFindItem(3, u): UnionFindItem(2, u)}, uf4.leaders)

    def test_find_leader_succeeds(self):
        uf = UnionFind()
        uf.add_singleton(2)
        uf.add_singleton(3)
        uf.add_singleton(4)
        uf.union(2, 3)
        uf.union(3, 4)
        self.assertEqual({UnionFindItem(2): UnionFindItem(2), UnionFindItem(3): UnionFindItem(2), UnionFindItem(4): UnionFindItem(3)}, uf.leaders)
        self.assertEqual(2, uf.find_leader(4))
        self.assertEqual({UnionFindItem(2): UnionFindItem(2), UnionFindItem(3): UnionFindItem(2), UnionFindItem(4): UnionFindItem(2)}, uf.leaders)
        self.assertEqual({UnionFindItem(2): frozenset([UnionFindItem(3), UnionFindItem(4)]), UnionFindItem(3): frozenset(), UnionFindItem(4): frozenset()}, uf.graph)

    def test_get_equivalence_class(self):
        uf = UnionFind()
        uf.add_singleton(2)
        uf.add_singleton(3)
        uf.add_singleton(4)
        uf.union(2, 3)
        uf.union(3, 4)
        uf.add_singleton(5)
        xs = uf.get_equivalence_class(4)
        self.assertEqual({2, 3, 4}, xs)
        self.assertEqual({5}, uf.get_equivalence_class(5))


if __name__ == '__main__':
    unittest.main()
