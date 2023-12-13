import expecttest
import pandas as pd

from schema import Schema, SchemaNode


class TestEx1(expecttest.TestCase):

    def initialise(self):
        s = Schema()

        # Load up the CSVs
        cardnum = pd.read_csv("./csv/bonuses/cardnum.csv").set_index("val_id")
        person = pd.read_csv("./csv/bonuses/person.csv").set_index("cardnum")

        # Insert them into the data frames
        s.insert_dataframe(cardnum, "cardnum")
        s.insert_dataframe(person, "person")

        # Get handles on the nodes now in the schema - may be slightly ugly but
        # we can slap on a load of syntactic sugar
        c_cardnum = SchemaNode("cardnum", cluster="cardnum")
        p_cardnum = SchemaNode("cardnum", cluster="person")

        s.blend(c_cardnum, p_cardnum, under="Cardnum")
        return s

        # SCHEMA:
        # val_id ---> cardnum ---> person

    # GOAL 1: Find all people who made trips
    # Get the cardnum for each trip, use it to infer who made the trip,
    # and then restrict it to valid trips
    def test_ex1_goal1_step1(self):
        s = self.initialise()
        t1 = s.get(["cardnum.cardnum"])
        self.assertExpectedInline(str(t1), """\
[cardnum.cardnum || ]
Empty DataFrame
Columns: []
Index: [5172, 2354, 1410, 1111, 4412]

""")
        # Get every cardnum in the cardnum csv
        # [cardnum.cardnum || ]
        #  5172
        #  2354
        #  1410
        #  1111
        #  4412

    def test_ex1_goal1_step2(self):
        s = self.initialise()
        t1 = s.get(["cardnum.cardnum"])
        t2 = t1.infer(["cardnum.cardnum"], "person.person")
        self.assertExpectedInline(str(t2), """\
[cardnum.cardnum || person.person]
                person.person
cardnum.cardnum              
1111.0                  Steve
1410.0                    Tom
2354.0                  Steve
2 keys hidden
2 values hidden

""")
        # Use this cardnum to infer who made the trip
        # Dropped keys! Values populate keys.
        # [cardnum.cardnum || person.person]
        #  2354            || Steve
        #  1410            || Tom
        #  1111            || Steve

    def test_ex1_goal1_step3(self):
        s = self.initialise()
        t1 = s.get(["cardnum.cardnum"])
        t2 = t1.infer(["cardnum.cardnum"], "person.cardnum")
        t3 = t2.infer(["person.cardnum"], "person.person")
        self.assertExpectedInline(str(t3), """\
[cardnum.cardnum || person.cardnum person.person]
                 person.cardnum person.person
cardnum.cardnum                              
1111.0                     1111         Steve
1410.0                     1410           Tom
2354.0                     2354         Steve
5172.0                     5172           NaN
4412.0                     4412           NaN
4 values hidden

""")

    def test_ex1_goal1_step4(self):
        s = self.initialise()
        t1 = s.get(["cardnum.cardnum"])
        t2 = t1.infer(["cardnum.cardnum"], "person.person")
        t3 = t2.compose(["cardnum.val_id"], "cardnum.cardnum")
        self.assertExpectedInline(str(t3), """\
[cardnum.val_id || person.person]
               person.person
cardnum.val_id              
2.0                    Steve
5.0                    Steve
3.0                      Tom
4.0                    Steve
2 keys hidden
2 values hidden

""")
        # Hey, I know how to get the val_id(s) given the cardnum
        # [cardnum.val_id || person.person]
        # 2               || Steve
        # 5               || Steve
        # 3               || Tom
        # 4               || Steve

    # But what if, as a byproduct, I want to know all people who have cardnums?
    # GOAL 2: Same as goal 1, but include people who didn't make trips
    # Get the cardnum for each PERSON, use it to infer who made the trip,
    # and then restrict it to valid trips

    def test_ex1_goal2_step1_get(self):
        s = self.initialise()
        t11 = s.get(["person.cardnum"])
        self.assertExpectedInline(str(t11), """\
[person.cardnum || ]
Empty DataFrame
Columns: []
Index: [1111, 1410, 2354, 6440, 5467]

""")
        # Get me every value of cardnum in person
        # [person.cardnum || ]
        #  1111
        #  1410
        #  2354
        #  6440
        #  5467

    def test_ex1_goal2_step2_infer(self):
        s = self.initialise()
        t11 = s.get(["person.cardnum"])
        t12 = t11.infer(["person.cardnum"], "person.person")
        self.assertExpectedInline(str(t12), """\
[person.cardnum || person.person]
               person.person
person.cardnum              
1111                   Steve
1410                     Tom
2354                   Steve
6440                   Harry
5467                    Dick

""")
        # From the value of cardnum, tell me who the card belongs to
        # [person.cardnum  || person.person]
        #  1111            || Steve
        #  1410            || Tom
        #  2354            || Steve
        #  6440            || Harry
        #  5467            || Dick

    # If that's all I want, I can stop here! But if I want to know which people
    # did NOT make trips, I can simply do

    def test_ex1_goal2_step3_composeAndSort(self):
        s = self.initialise()
        t11 = s.get(["person.cardnum"])
        t12 = t11.infer(["person.cardnum"], "person.person")
        t13 = t12.compose(["cardnum.val_id"], "person.cardnum")
        t14 = t13.sort(["person.person"])
        self.assertExpectedInline(str(t14), """\
[cardnum.val_id || person.person]
               person.person
cardnum.val_id              
2.0                    Steve
5.0                    Steve
4.0                    Steve
3.0                      Tom
2 keys hidden
2 values hidden

""")
        # I have a mapping from val_id to cardnum
        # [cardnum.val_id || person.person]
        # 2               || Steve
        # 5               || Steve
        # 3               || Tom
        # 4               || Steve

    def test_ex1_goal3_step1_getAndInferWithHiddenKey(self):
        s = self.initialise()
        t21 = s.get(["cardnum.cardnum"])
        t22 = t21.infer(["cardnum.cardnum"], "cardnum.val_id")
        self.assertExpectedInline(str(t22), """\
[cardnum.cardnum || cardnum.val_id]
                cardnum.val_id
cardnum.cardnum               
5172                       [1]
2354                    [2, 5]
1410                       [3]
1111                       [4]
4412                       [8]

""")
        # Get every cardnum in the cardnum csv
        # [cardnum.cardnum || ]
        #  5172
        #  2354
        #  1410
        #  1111
        #  4412

    def test_ex1_goal3_step2_assignAndPlus(self):
        s = self.initialise()
        t21 = s.get(["cardnum.cardnum"])
        t22 = t21.infer(["cardnum.cardnum"], "cardnum.val_id")
        t23 = t22.assign("cardnum.plusone", t22["cardnum.cardnum"] + t22["cardnum.val_id"])
        self.assertExpectedInline(str(t23), """\
[cardnum.cardnum || cardnum.val_id cardnum.plusone]
                cardnum.val_id cardnum.plusone
cardnum.cardnum                               
5172                       [1]          [5173]
2354                    [2, 5]    [2356, 2359]
1410                       [3]          [1413]
1111                       [4]          [1115]
4412                       [8]          [4420]

""")

    def test_ex1_goal3_step3_setKey(self):
        s = self.initialise()
        t21 = s.get(["cardnum.cardnum"])
        t22 = t21.infer(["cardnum.cardnum"], "cardnum.val_id")
        t23 = t22.assign("cardnum.plusone", t22["cardnum.cardnum"] + t22["cardnum.val_id"])
        t24 = t23.set_key(["cardnum.plusone"])

        self.assertExpectedInline(str(t24), """\
[cardnum.plusone || cardnum.cardnum cardnum.val_id]
                cardnum.cardnum cardnum.val_id
cardnum.plusone                               
5173                     [5172]            [1]
2356                     [2354]            [2]
2359                     [2354]            [5]
1413                     [1410]            [3]
1115                     [1111]            [4]
4420                     [4412]            [8]

""")

    def test_ex1_goal4_step1_get(self):
        s = self.initialise()
        t31 = s.get(['cardnum.val_id'])
        self.assertExpectedInline(str(t31), """\
[cardnum.val_id || ]
Empty DataFrame
Columns: []
Index: [1, 2, 3, 4, 5, 8]

""")

    def test_ex1_goal4_step2_infer(self):
        s = self.initialise()
        t31 = s.get(['cardnum.val_id'])
        t32 = t31.infer(['cardnum.val_id'], 'cardnum.cardnum')
        self.assertExpectedInline(str(t32), """\
[cardnum.val_id || cardnum.cardnum]
                cardnum.cardnum
cardnum.val_id                 
1                          5172
2                          2354
3                          1410
4                          1111
5                          2354
8                          4412

""")

    def test_ex1_goal4_step3_inferChain(self):
        s = self.initialise()
        t31 = s.get(['cardnum.val_id'])
        t32 = t31.infer(['cardnum.val_id'], 'cardnum.cardnum')
        t33 = t32.infer(['cardnum.cardnum'], 'person.cardnum')
        self.assertExpectedInline(str(t33), """\
[cardnum.val_id || cardnum.cardnum person.cardnum]
                cardnum.cardnum  person.cardnum
cardnum.val_id                                 
1                          5172            5172
2                          2354            2354
3                          1410            1410
4                          1111            1111
5                          2354            2354
8                          4412            4412

""")

    def test_ex1_goal4_step4_inferChainAgain(self):
        s = self.initialise()
        t31 = s.get(['cardnum.val_id'])
        t32 = t31.infer(['cardnum.val_id'], 'cardnum.cardnum')
        t33 = t32.infer(['cardnum.cardnum'], 'person.cardnum')
        t34 = t33.infer(['person.cardnum'], 'person.person')
        self.assertExpectedInline(str(t34), """\
[cardnum.val_id || cardnum.cardnum person.cardnum person.person]
                cardnum.cardnum  person.cardnum person.person
cardnum.val_id                                               
4.0                      1111.0            1111         Steve
3.0                      1410.0            1410           Tom
2.0                      2354.0            2354         Steve
5.0                      2354.0            2354         Steve
1.0                      5172.0            5172           NaN
8.0                      4412.0            4412           NaN
4 values hidden

""")

    def test_ex1_goal4_step5_hide(self):
        s = self.initialise()
        t31 = s.get(['cardnum.val_id'])
        t32 = t31.infer(['cardnum.val_id'], 'cardnum.cardnum')
        t33 = t32.infer(['cardnum.cardnum'], 'person.cardnum')
        t34 = t33.infer(['person.cardnum'], 'person.person')
        t35 = t34.hide('person.cardnum').hide('cardnum.cardnum')
        self.assertExpectedInline(str(t35), """\
[cardnum.val_id || person.person]
               person.person
cardnum.val_id              
4.0                    Steve
3.0                      Tom
2.0                    Steve
5.0                    Steve
2 keys hidden
2 values hidden

""")

    def test_ex1_goal4_step6_showAfterHide(self):
        s = self.initialise()
        t31 = s.get(['cardnum.val_id'])
        t32 = t31.infer(['cardnum.val_id'], 'cardnum.cardnum')
        t33 = t32.infer(['cardnum.cardnum'], 'person.cardnum')
        t34 = t33.infer(['person.cardnum'], 'person.person')
        t35 = t34.hide('person.cardnum').hide('cardnum.cardnum')
        t36 = t35.show('person.cardnum').show('cardnum.cardnum')
        self.assertExpectedInline(str(t36), """\
[cardnum.val_id || cardnum.cardnum person.cardnum person.person]
                cardnum.cardnum  person.cardnum person.person
cardnum.val_id                                               
4.0                      1111.0            1111         Steve
3.0                      1410.0            1410           Tom
2.0                      2354.0            2354         Steve
5.0                      2354.0            2354         Steve
1.0                      5172.0            5172           NaN
8.0                      4412.0            4412           NaN
4 values hidden

""")

    def test_ex1_goal5_step1_getAndCompose(self):
        s = self.initialise()
        t41 = (s.get(['person.cardnum']).infer(['person.cardnum'], 'person.person'))
        t42 = t41.compose(['cardnum.val_id'], 'person.cardnum')
        self.assertExpectedInline(str(t42), """\
[cardnum.val_id || person.person]
               person.person
cardnum.val_id              
2.0                    Steve
5.0                    Steve
3.0                      Tom
4.0                    Steve
2 keys hidden
2 values hidden

""")
    def test_ex1_goal5_step2_setKey(self):
        s = self.initialise()
        t41 = (s.get(['person.cardnum']).infer(['person.cardnum'], 'person.person'))
        t42 = t41.compose(['cardnum.val_id'], 'person.cardnum')
        t43 = t42.set_key(["person.person"])
        self.assertExpectedInline(str(t43), """\
[person.person || cardnum.val_id]
                cardnum.val_id
person.person                 
Steve          [2.0, 5.0, 4.0]
Tom                      [3.0]
3 keys hidden

""")