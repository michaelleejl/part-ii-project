import unittest

import pandas as pd

from schema.schema import Schema


class TestSchema(unittest.TestCase):
    def test_schema_insert(self):
        test = pd.read_csv("./test_schema.csv").dropna()
        test = test.set_index(["trip_id", "cardnum"])
        schema = Schema()
        schema.insert(test, "test_schema")
        print(schema)

    def test_schema_blend(self):
        bonus = pd.read_csv("./bonus.csv").dropna().set_index(["trip_id", "cardnum"])
        person = pd.read_csv("./person.csv").dropna().set_index(["cardnum"])
        schema = Schema()
        schema.insert(bonus, "bonus")
        schema.insert(person, "person")
        cardnum_bonus = schema.get_node("cardnum", "bonus")
        cardnum_person = schema.get_node("cardnum", "person")
        equivalence = pd.DataFrame({cardnum_bonus: [101, 111], cardnum_person: [101, 111]})
        schema.blend(cardnum_bonus, cardnum_person, "Cardnum", equivalence)
        print(schema)