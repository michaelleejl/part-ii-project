import expecttest

import pandas as pd

from schema import SchemaNode
from schema.schema import Schema


class TestSchema(expecttest.TestCase):
    def test_schema_insert(self):
        test = pd.read_csv("./test_schema.csv").dropna()
        test = test.set_index(["trip_id", "cardnum"])
        schema = Schema()
        schema.insert_dataframe(test, "test_schema")
        self.assertExpectedInline(str(schema), """\
                                                FULLY CONNECTED CLUSTERS 
                                                ==========================
                                                
                                                ==========================
                                                test_schema
                                                --------------------------
                                                test_schema.(trip_id)
                                                test_schema.(cardnum)
                                                test_schema.(person)
                                                test_schema.(bonus)
                                                ==========================
                                                
                                                ==========================
                                                
                                                ADJACENCY LIST 
                                                ==========================
                                                
                                                
                                                ==========================
                                                
                                                EQUIVALENCE CLASSES 
                                                ==========================
                                                
                                                ==========================
                                                Class 0
                                                --------------------------
                                                test_schema.(trip_id)
                                                ==========================
                                                
                                                ==========================
                                                Class 1
                                                --------------------------
                                                test_schema.(cardnum)
                                                ==========================
                                                
                                                ==========================
                                                Class 2
                                                --------------------------
                                                test_schema.(person)
                                                ==========================
                                                
                                                ==========================
                                                Class 3
                                                --------------------------
                                                test_schema.(bonus)
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
        schema.blend(cardnum_bonus, cardnum_person)
        self.assertExpectedInline(str(schema), """\
                                                FULLY CONNECTED CLUSTERS 
                                                ==========================
                                                
                                                ==========================
                                                bonus
                                                --------------------------
                                                bonus.(trip_id)
                                                bonus.(cardnum)
                                                bonus.(bonus)
                                                ==========================
                                                
                                                ==========================
                                                person
                                                --------------------------
                                                person.(cardnum)
                                                person.(person)
                                                ==========================
                                                
                                                ==========================
                                                
                                                ADJACENCY LIST 
                                                ==========================
                                                
                                                
                                                ==========================
                                                
                                                EQUIVALENCE CLASSES 
                                                ==========================
                                                
                                                ==========================
                                                Class 0
                                                --------------------------
                                                bonus.(trip_id)
                                                ==========================
                                                
                                                ==========================
                                                Class 1
                                                --------------------------
                                                bonus.(cardnum)
                                                person.(cardnum)
                                                ==========================
                                                
                                                ==========================
                                                Class 2
                                                --------------------------
                                                bonus.(bonus)
                                                ==========================
                                                
                                                ==========================
                                                Class 3
                                                --------------------------
                                                person.(person)
                                                ==========================
                                                
                                                ==========================
                                                """)
    def test_schema_cloneWithoutSpecifiedName_succeeds(self):
        s = Schema()
        u = SchemaNode("u", cluster="cluster")
        s.add_node("u", "cluster")
        s.clone(u)
        self.assertExpectedInline(str(s), """\
FULLY CONNECTED CLUSTERS 
==========================


==========================

ADJACENCY LIST 
==========================


==========================

EQUIVALENCE CLASSES 
==========================

==========================
Class 0
--------------------------
cluster.u_1
cluster.u
==========================

==========================
""")

    def test_schema_cloneWithSpecifiedName_succeeds(self):
        s = Schema()
        u = SchemaNode("u", cluster="cluster")
        s.add_node("u", "cluster")
        s.add_node("v", "cluster")
        s.clone(u, "v")
        self.assertExpectedInline(str(s), """\
FULLY CONNECTED CLUSTERS 
==========================


==========================

ADJACENCY LIST 
==========================


==========================

EQUIVALENCE CLASSES 
==========================

==========================
Class 0
--------------------------
cluster.v_1
cluster.u
==========================

==========================
Class 1
--------------------------
cluster.v
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
        schema.blend(cardnum_bonus, cardnum_person)
        t = schema.get([cardnum_person, cardnum_bonus])
        self.assertExpectedInline(str(t), """[person.cardnum bonus.cardnum || ]""")