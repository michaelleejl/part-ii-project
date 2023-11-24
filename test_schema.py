import unittest

import numpy as np
import pandas as pd
from schema import invert_cardinality, Cardinality, KeyDuplicationException, check_for_duplicate_keys, SchemaNode, \
                   determine_cardinality, SchemaEdge, SchemaEdgeList, SchemaGraph, Schema, compute_equivalence_classes, \
                   get_unrelated


class SchemaTest(unittest.TestCase):

    def test_schema(self):
        test = pd.read_csv("schema/tests/test_schema.csv").dropna()
        test = test.set_index(["trip_id", "cardnum"])
        schema = Schema()
        schema.insert(test, "test_schema")
        print(schema)


if __name__ == '__main__':
    unittest.main()
