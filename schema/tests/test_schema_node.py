import unittest

import pandas as pd

from schema import SchemaNode


class TestSchemaNode(unittest.TestCase):
    def test_schemaNodeEquality_returnsTrue_ifNameAndFamilyEqual(self):
        u = SchemaNode("name", "1")
        v = SchemaNode("name", "1")
        self.assertEqual(u, v)

    def test_schemaNodeEquality_returnsFalse_ifFamilyNotEqual(self):
        u = SchemaNode("name", family="1")
        v = SchemaNode("name", family="2")
        self.assertNotEqual(u, v)

    def test_schemaNodeEquality_returnsFalse_ifNameNotEqual(self):
        u = SchemaNode("name", family="1")
        v = SchemaNode("name'", family="1")
        self.assertNotEqual(u, v)