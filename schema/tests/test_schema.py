import expecttest

import pandas as pd
from schema.schema import Schema


class TestSchema(expecttest.TestCase):
    # [cardnum trip_id || bonus t_start]

    def test_schema_insert(self):
        test_df = pd.read_csv("./schema/tests/test_schema.csv").dropna()
        test_df = test_df.set_index(["trip_id", "cardnum"])
        schema = Schema()
        test = schema.insert_dataframe(test_df)
        test["trip_id"].id_prefix = 0
        test["cardnum"].id_prefix = 0
        test["person"].id_prefix = 0
        test["bonus"].id_prefix = 0
        self.assertExpectedInline(str(schema), """\
ADJACENCY LIST 
==========================

==========================
trip_id;cardnum
--------------------------
trip_id;cardnum ---> bonus
trip_id;cardnum ---> trip_id
trip_id;cardnum ---> cardnum
trip_id;cardnum ---> person
==========================

==========================
trip_id
--------------------------
trip_id;cardnum ---> trip_id
==========================

==========================
cardnum
--------------------------
trip_id;cardnum ---> cardnum
==========================

==========================
person
--------------------------
trip_id;cardnum ---> person
==========================

==========================
bonus
--------------------------
trip_id;cardnum ---> bonus
==========================

==========================

EQUIVALENCE CLASSES 
==========================

==========================
Class 0
--------------------------
trip_id
==========================

==========================
Class 1
--------------------------
cardnum
==========================

==========================
Class 2
--------------------------
person
==========================

==========================
Class 3
--------------------------
bonus
==========================

==========================
""")

    def test_schema_blend(self):
        bonus_df = pd.read_csv("./schema/tests/bonus.csv").dropna().set_index(["trip_id", "cardnum"])
        person_df = pd.read_csv("./schema/tests/person.csv").dropna().set_index(["cardnum"])
        schema = Schema()
        bonus = schema.insert_dataframe(bonus_df)
        person = schema.insert_dataframe(person_df)
        bonus["trip_id"].id_prefix = 0
        bonus["cardnum"].id_prefix = 0
        bonus["bonus"].id_prefix = 0
        person["person"].id_prefix = 0
        person["cardnum"].id_prefix = 0
        cardnum_bonus = bonus["cardnum"]
        cardnum_person = person["cardnum"]
        Cardnum = schema.create_class("Cardnum")
        schema.blend(cardnum_bonus, cardnum_person, Cardnum)
        self.assertExpectedInline(str(schema), """\
ADJACENCY LIST 
==========================

==========================
trip_id;cardnum
--------------------------
trip_id;cardnum ---> cardnum
trip_id;cardnum ---> bonus
trip_id;cardnum ---> trip_id
==========================

==========================
trip_id
--------------------------
trip_id;cardnum ---> trip_id
==========================

==========================
cardnum
--------------------------
trip_id;cardnum ---> cardnum
==========================

==========================
bonus
--------------------------
trip_id;cardnum ---> bonus
==========================

==========================
cardnum
--------------------------
cardnum ---> person
==========================

==========================
person
--------------------------
cardnum ---> person
==========================

==========================

EQUIVALENCE CLASSES 
==========================

==========================
Class 0
--------------------------
trip_id
==========================

==========================
Class 1
--------------------------
Cardnum
cardnum
cardnum
==========================

==========================
Class 2
--------------------------
bonus
==========================

==========================
Class 3
--------------------------
person
==========================

==========================
""")

    def test_schema_get(self):
        bonus_df = pd.read_csv("./schema/tests/bonus.csv").dropna().set_index(["trip_id", "cardnum"])
        person_df = pd.read_csv("./schema/tests/person.csv").dropna().set_index(["cardnum"])
        schema = Schema()
        bonus = schema.insert_dataframe(bonus_df)
        person = schema.insert_dataframe(person_df)
        cardnum_bonus = bonus["cardnum"]
        cardnum_person = person["cardnum"]
        Cardnum = schema.create_class("Cardnum")
        schema.blend(cardnum_bonus, cardnum_person, Cardnum)
        t = schema.get([cardnum_person, cardnum_bonus])
        self.assertExpectedInline(str(t), """\
[cardnum cardnum_1 || ]
Empty DataFrame
Columns: []
Index: [(101, 101), (101, 111), (101, 100), (111, 101), (111, 111), (111, 100)]

""")