import unittest

import pandas as pd

from schema import SchemaNode
from union_find.union_find import UnionFindItem, UnionFind


class UnionFindTest(unittest.TestCase):
    def test_item_equality_isValueEquality(self):
        i1 = UnionFindItem(2)
        i2 = UnionFindItem(2)
        i3 = UnionFindItem(3)
        self.assertEqual(i1, i2)
        self.assertNotEqual(i1, i3)

    def test_add_singleton_succeedsWithNoSideEffects_whenSingletonNotInUF(self):
        uf = UnionFind.initialise()
        uf2 = UnionFind.add_singleton(uf, 2)
        self.assertEqual({UnionFindItem(2): UnionFindItem(2)}, uf2.leaders)
        self.assertEqual({UnionFindItem(2): 0}, uf2.rank)
        self.assertEqual({UnionFindItem(2): frozenset()}, uf2.graph)
        self.assertEqual({}, uf.leaders)

    def test_add_singleton_isIdempotent(self):
        uf = UnionFind.initialise()
        uf2 = UnionFind.add_singleton(uf, 2)
        uf3 = UnionFind.add_singleton(uf2, 2)
        self.assertEqual({UnionFindItem(2): UnionFindItem(2)}, uf3.leaders)
        self.assertEqual({UnionFindItem(2): 0}, uf3.rank)
        self.assertEqual({UnionFindItem(2): frozenset()}, uf3.graph)
        self.assertEqual({}, uf.leaders)

    def test_union_succeeds(self):
        uf = UnionFind.initialise()
        uf2 = UnionFind.add_singleton(uf, 2)
        uf3 = UnionFind.add_singleton(uf2, 3)
        uf4 = UnionFind.union(uf3, 2, 3)
        self.assertEqual({UnionFindItem(2): UnionFindItem(2), UnionFindItem(3): UnionFindItem(2)}, uf4.leaders)

    def test_find_leader_succeeds(self):
        uf = UnionFind.initialise()
        uf = UnionFind.add_singleton(uf, 2)
        uf = UnionFind.add_singleton(uf, 3)
        uf = UnionFind.add_singleton(uf, 4)
        uf = UnionFind.union(uf, 2, 3)
        uf = UnionFind.union(uf, 3, 4)
        self.assertEqual({UnionFindItem(2): UnionFindItem(2), UnionFindItem(3): UnionFindItem(2), UnionFindItem(4): UnionFindItem(3)}, uf.leaders)
        self.assertEqual(2, uf.find_leader(4))
        self.assertEqual({UnionFindItem(2): UnionFindItem(2), UnionFindItem(3): UnionFindItem(2), UnionFindItem(4): UnionFindItem(2)}, uf.leaders)
        self.assertEqual({UnionFindItem(2): frozenset([UnionFindItem(3), UnionFindItem(4)]), UnionFindItem(3): frozenset(), UnionFindItem(4): frozenset()}, uf.graph)

    def test_get_equivalence_class(self):
        uf = UnionFind.initialise()
        uf = UnionFind.add_singleton(uf, 2)
        uf = UnionFind.add_singleton(uf, 3)
        uf = UnionFind.add_singleton(uf, 4)
        uf = UnionFind.union(uf, 2, 3)
        uf = UnionFind.union(uf, 3, 4)
        uf = UnionFind.add_singleton(uf, 5)
        xs = uf.get_equivalence_class(4)
        self.assertEqual({2, 3, 4}, xs)
        self.assertEqual({5}, uf.get_equivalence_class(5))


if __name__ == '__main__':
    unittest.main()
