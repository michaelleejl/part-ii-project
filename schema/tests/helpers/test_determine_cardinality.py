import unittest

import pandas as pd

from schema.cardinality import Cardinality
from backend.helpers.determine_cardinality import determine_cardinality


class TestDetermineCardinality(unittest.TestCase):
    def test_determine_cardinality_returnsOneToOne_ifOneToOne(self):
        mapping = pd.DataFrame({"l": [0, 1, 2], "r": [0, 1, 2]})
        self.assertEqual(determine_cardinality(mapping, "l", "r"), Cardinality.ONE_TO_ONE)

    def test_determine_cardinality_returnsOneToMany_ifOneToMany(self):
        mapping = pd.DataFrame({"l": [0, 1, 1], "r": [0, 1, 2]})
        self.assertEqual(determine_cardinality(mapping, "l", "r"), Cardinality.ONE_TO_MANY)

    def test_determine_cardinality_returnsManyToOne_ifManyToOne(self):
        mapping = pd.DataFrame({"l": [0, 1, 2], "r": [0, 1, 1]})
        self.assertEqual(determine_cardinality(mapping, "l", "r"), Cardinality.MANY_TO_ONE)

    def test_determine_cardinality_returnsManyToMany_ifManyToMany(self):
        mapping = pd.DataFrame({"l": [0, 1, 1, 2], "r": [0, 1, 2, 2]})
        self.assertEqual(determine_cardinality(mapping, "l", "r"), Cardinality.MANY_TO_MANY)