import unittest

from schema.cardinality import Cardinality
from schema.helpers.invert_cardinality import invert_cardinality


class TestInvertCardinality(unittest.TestCase):
    def test_invert_cardinality(self):
        self.assertEqual(invert_cardinality(Cardinality.ONE_TO_ONE), Cardinality.ONE_TO_ONE)
        self.assertEqual(invert_cardinality(Cardinality.ONE_TO_MANY), Cardinality.MANY_TO_ONE)
        self.assertEqual(invert_cardinality(Cardinality.MANY_TO_ONE), Cardinality.ONE_TO_MANY)
        self.assertEqual(invert_cardinality(Cardinality.MANY_TO_MANY), Cardinality.MANY_TO_MANY)