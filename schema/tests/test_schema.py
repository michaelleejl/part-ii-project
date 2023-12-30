import expecttest

import pandas as pd

from schema.node import SchemaNode
from schema.schema import Schema


class TestSchema(expecttest.TestCase):
    # [cardnum trip_id || bonus t_start]

    def test_schema_insert(self):
        test = pd.read_csv("./test_schema.csv").dropna()
        test = test.set_index(["trip_id", "cardnum"])
        schema = Schema()
        schema.insert_dataframe(test)
        self.assertExpectedInline(str(schema), """\
ADJACENCY LIST 
==========================

==========================
test_schema.trip_id;cardnum
--------------------------
trip_id;cardnum ---> person
trip_id;cardnum ---> trip_id
trip_id;cardnum ---> bonus
trip_id;cardnum ---> cardnum
==========================

==========================
test_schema.trip_id
--------------------------
trip_id;cardnum ---> trip_id
==========================

==========================
test_schema.cardnum
--------------------------
trip_id;cardnum ---> cardnum
==========================

==========================
test_schema.person
--------------------------
trip_id;cardnum ---> person
==========================

==========================
test_schema.bonus
--------------------------
trip_id;cardnum ---> bonus
==========================

==========================

EQUIVALENCE CLASSES 
==========================

==========================
Class 0
--------------------------
test_schema.trip_id
==========================

==========================
Class 1
--------------------------
test_schema.cardnum
==========================

==========================
Class 2
--------------------------
test_schema.person
==========================

==========================
Class 3
--------------------------
test_schema.bonus
==========================

==========================
""")

    def test_schema_blend(self):
        bonus = pd.read_csv("./bonus.csv").dropna().set_index(["trip_id", "cardnum"])
        person = pd.read_csv("./person.csv").dropna().set_index(["cardnum"])
        schema = Schema()
        schema.insert_dataframe(bonus, "bonus")
        schema.insert_dataframe(person, "person")
        cardnum_bonus = SchemaNode("cardnum", cluster="bonus")
        cardnum_person = SchemaNode("cardnum", cluster="person")
        schema.blend(cardnum_bonus, cardnum_person, "Cardnum")
        self.assertExpectedInline(str(schema), """\
ADJACENCY LIST 
==========================

==========================
bonus.trip_id;cardnum
--------------------------
trip_id;cardnum ---> cardnum
trip_id;cardnum ---> bonus
trip_id;cardnum ---> trip_id
==========================

==========================
bonus.trip_id
--------------------------
trip_id;cardnum ---> trip_id
==========================

==========================
bonus.cardnum
--------------------------
trip_id;cardnum ---> cardnum
==========================

==========================
bonus.bonus
--------------------------
trip_id;cardnum ---> bonus
==========================

==========================
person.cardnum
--------------------------
cardnum ---> person
==========================

==========================
person.person
--------------------------
cardnum ---> person
==========================

==========================

EQUIVALENCE CLASSES 
==========================

==========================
Class 0
--------------------------
bonus.trip_id
==========================

==========================
Class 1
--------------------------
bonus.cardnum
Cardnum
person.cardnum
==========================

==========================
Class 2
--------------------------
bonus.bonus
==========================

==========================
Class 3
--------------------------
person.person
==========================

==========================
""")

    def test_schema_get(self):
        bonus = pd.read_csv("./bonus.csv").dropna().set_index(["trip_id", "cardnum"])
        person = pd.read_csv("./person.csv").dropna().set_index(["cardnum"])
        schema = Schema()
        schema.insert_dataframe(bonus, "bonus")
        schema.insert_dataframe(person, "person")
        cardnum_bonus = SchemaNode("cardnum", cluster="bonus")
        cardnum_person = SchemaNode("cardnum", cluster="person")
        schema.blend(cardnum_bonus, cardnum_person, "Cardnum")
        t = schema.get(["person.cardnum", "bonus.cardnum"])
        self.assertExpectedInline(str(t), """\
[person.cardnum bonus.cardnum || ]
Empty DataFrame
Columns: []
Index: [(101, 101), (101, 111), (101, 100), (111, 101), (111, 111), (111, 100)]

""")