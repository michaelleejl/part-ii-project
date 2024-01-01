import expecttest
import pandas as pd

from schema import Schema, SchemaNode


class TestEx1(expecttest.TestCase):

    def initialise(self):
        s = Schema()

        # Load up the CSVs
        cardnum_df = pd.read_csv("./csv/bonuses/cardnum.csv").set_index("val_id")
        person_df = pd.read_csv("./csv/bonuses/person.csv").set_index("cardnum")

        # Insert them into the data frames
        cardnum = s.insert_dataframe(cardnum_df)
        person = s.insert_dataframe(person_df)

        # Get handles on the nodes now in the schema - may be slightly ugly but
        # we can slap on a load of syntactic sugar
        c_cardnum = cardnum["cardnum"]
        p_cardnum = person["cardnum"]

        Cardnum = s.create_class("Cardnum")

        s.blend(c_cardnum, p_cardnum, Cardnum)
        return s, cardnum, person

        # SCHEMA:
        # val_id ---> cardnum ---> person

    # GOAL 1: Find all people who made trips
    # Get the cardnum for each trip, use it to infer who made the trip,
    # and then restrict it to valid trips
    def test_ex1_goal1_step1(self):
        s, cardnum, person = self.initialise()
        c_cardnum = cardnum["cardnum"]
        t1 = s.get([c_cardnum])
        self.assertExpectedInline(str(t1), """\
[cardnum || ]
Empty DataFrame
Columns: []
Index: []

""")
        # Get every cardnum in the cardnum csv
        # [cardnum.cardnum || ]
        #  5172
        #  2354
        #  1410
        #  1111
        #  4412

    def test_ex1_goal1_step2(self):
        s, cardnum, person = self.initialise()
        t1 = s.get([cardnum["cardnum"]])
        t2 = t1.infer(["cardnum"], person["person"]).sort("cardnum")
        self.assertExpectedInline(str(t2), """\
[cardnum || person]
        person
cardnum       
1111     Steve
1410       Tom
2354     Steve

""")
        # Use this cardnum to infer who made the trip
        # Dropped keys! Values populate keys.
        # [cardnum.cardnum || person.person]
        #  2354            || Steve
        #  1410            || Tom
        #  1111            || Steve

    def test_ex1_goal1_step3(self):
        s, cardnum, person = self.initialise()
        t1 = s.get([cardnum["cardnum"]])
        t2 = t1.infer(["cardnum"], person["cardnum"])
        t3 = t2.infer(["cardnum_1"], person["person"]).sort("cardnum")
        self.maxDiff = None
        self.assertExpectedInline(str(t3), """\
[cardnum || cardnum_1 person]
         cardnum_1 person
cardnum                  
1111.0        1111  Steve
1410.0        1410    Tom
2354.0        2354  Steve
4412.0        4412    NaN
5172.0        5172    NaN

""")

    def test_ex1_goal1_step4(self):
        s, cardnum, person = self.initialise()
        t1 = s.get([cardnum["cardnum"]])
        t2 = t1.infer(["cardnum"], person["person"])
        t3 = t2.compose([cardnum["val_id"]], "cardnum")
        self.assertExpectedInline(str(t3), """\
[val_id || person]
       person
val_id       
2       Steve
3         Tom
4       Steve
5       Steve
2 keys hidden

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
        s, cardnum, person = self.initialise()
        t11 = s.get([person["cardnum"]])
        self.assertExpectedInline(str(t11), """\
[cardnum || ]
Empty DataFrame
Columns: []
Index: []

""")
        # Get me every value of cardnum in person
        # [person.cardnum || ]
        #  1111
        #  1410
        #  2354
        #  6440
        #  5467

    def test_ex1_goal2_step2_infer(self):
        s, cardnum, person = self.initialise()
        t11 = s.get([person["cardnum"]])
        t12 = t11.infer(["cardnum"], person["person"]).sort(["cardnum"])
        self.assertExpectedInline(str(t12), """\
[cardnum || person]
        person
cardnum       
1111     Steve
1410       Tom
2354     Steve
5467      Dick
6440     Harry

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
        s, cardnum, person = self.initialise()
        t11 = s.get([person["cardnum"]])
        t12 = t11.infer(["cardnum"], person["person"])
        t13 = t12.compose([cardnum["val_id"]], "cardnum")
        t14 = t13.sort(["person"])
        self.assertExpectedInline(str(t14), """\
[val_id || person]
       person
val_id       
2.0     Steve
4.0     Steve
5.0     Steve
3.0       Tom

""")
        # I have a mapping from val_id to cardnum
        # [cardnum.val_id || person.person]
        # 2               || Steve
        # 5               || Steve
        # 3               || Tom
        # 4               || Steve

    def test_ex1_goal3_step1_getAndInferWithHiddenKey(self):
        s, cardnum, person = self.initialise()
        t21 = s.get([cardnum["cardnum"]])
        t22 = t21.infer(["cardnum"], cardnum["val_id"]).sort(["val_id"])
        self.assertExpectedInline(str(t22), """\
[cardnum || val_id]
         val_id
cardnum        
1111        [4]
1410        [3]
2354     [2, 5]
4412        [8]
5172        [1]

""")
        # Get every cardnum in the cardnum csv
        # [cardnum.cardnum || ]
        #  5172
        #  2354
        #  1410
        #  1111
        #  4412

    def test_ex1_goal3_step2_assignAndPlus(self):
        s, cardnum, person = self.initialise()
        t21 = s.get([cardnum["cardnum"]])
        t22 = t21.infer(["cardnum"], cardnum["val_id"])
        t23 = t22.deduce(t22["cardnum"] + t22["val_id"], "numplusvalid")
        self.maxDiff = None
        self.assertExpectedInline(str(t23), """\
[cardnum || val_id numplusvalid]
         val_id  numplusvalid
cardnum                      
5172        [1]        [5173]
2354     [2, 5]  [2356, 2359]
1410        [3]        [1413]
1111        [4]        [1115]
4412        [8]        [4420]

""")

    def test_ex1_goal3_step3_setKey(self):
        s, cardnum, person = self.initialise()
        t21 = s.get([cardnum["cardnum"]])
        t22 = t21.infer(["cardnum"], cardnum["val_id"])
        t23 = t22.deduce(t22["cardnum"] + t22["val_id"], "numplusvalid")
        t24 = t23.set_key(["numplusvalid"])

        self.assertExpectedInline(str(t24), """\
[numplusvalid || cardnum val_id]
             cardnum val_id
numplusvalid               
5173          [5172]    [1]
2356          [2354]    [2]
2359          [2354]    [5]
1413          [1410]    [3]
1115          [1111]    [4]
4420          [4412]    [8]

""")

    def test_ex1_goal4_step1_get(self):
        s, cardnum, person = self.initialise()
        t31 = s.get([cardnum['val_id']])
        self.assertExpectedInline(str(t31), """\
[val_id || ]
Empty DataFrame
Columns: []
Index: []

""")

    def test_ex1_goal4_step2_infer(self):
        s, cardnum, person = self.initialise()
        t31 = s.get([cardnum['val_id']])
        t32 = t31.infer(['val_id'], cardnum['cardnum'])
        self.assertExpectedInline(str(t32), """\
[val_id || cardnum]
        cardnum
val_id         
1          5172
2          2354
3          1410
4          1111
5          2354
8          4412

""")

    def test_ex1_goal4_step3_inferChain(self):
        s, cardnum, person = self.initialise()
        t31 = s.get([cardnum['val_id']])
        t32 = t31.infer(['val_id'], cardnum['cardnum'])
        t33 = t32.infer(['cardnum'], person['cardnum'], with_name="person.cardnum")
        self.assertExpectedInline(str(t33), """\
[val_id || cardnum person.cardnum]
        cardnum  person.cardnum
val_id                         
1          5172            5172
2          2354            2354
5          2354            2354
3          1410            1410
4          1111            1111
8          4412            4412

""")

    def test_ex1_goal4_step4_inferChainAgain(self):
        s, cardnum, person = self.initialise()
        t31 = s.get([cardnum['val_id']])
        t32 = t31.infer(['val_id'], cardnum['cardnum'])
        t33 = t32.infer(['cardnum'], person['cardnum'], with_name="person.cardnum")
        t34 = t33.infer(['person.cardnum'], person['person'])
        self.maxDiff = None
        self.assertExpectedInline(str(t34), """\
[val_id || cardnum person.cardnum person]
        cardnum  person.cardnum person
val_id                                
4.0      1111.0          1111.0  Steve
3.0      1410.0          1410.0    Tom
2.0      2354.0          2354.0  Steve
5.0      2354.0          2354.0  Steve
1.0      5172.0          5172.0    NaN
8.0      4412.0          4412.0    NaN
2 values hidden

""")

    def test_ex1_goal4_step5_hide(self):
        s, cardnum, person = self.initialise()
        t31 = s.get([cardnum['val_id']])
        t32 = t31.infer(['val_id'], cardnum['cardnum'])
        t33 = t32.infer(['cardnum'], person['cardnum'], with_name="person.cardnum")
        t34 = t33.infer(['person.cardnum'], person['person'])
        t35 = t34.hide('person.cardnum').hide('cardnum')
        self.assertExpectedInline(str(t35), """\
[val_id || person]
       person
val_id       
4.0     Steve
3.0       Tom
2.0     Steve
5.0     Steve
2 keys hidden
2 values hidden

""")

    def test_ex1_goal4_step6_showAfterHide(self):
        s, cardnum, person = self.initialise()
        t31 = s.get([cardnum['val_id']])
        t32 = t31.infer(['val_id'], cardnum['cardnum'])
        t33 = t32.infer(['cardnum'], person['cardnum'], with_name="person.cardnum")
        t34 = t33.infer(['person.cardnum'], person['person'])
        t35 = t34.hide('person.cardnum').hide('cardnum')
        t36 = t35.show('person.cardnum').show('cardnum')
        self.maxDiff = None
        self.assertExpectedInline(str(t36), """\
[val_id || cardnum person.cardnum person]
        cardnum  person.cardnum person
val_id                                
4.0      1111.0          1111.0  Steve
3.0      1410.0          1410.0    Tom
2.0      2354.0          2354.0  Steve
5.0      2354.0          2354.0  Steve
1.0      5172.0          5172.0    NaN
8.0      4412.0          4412.0    NaN
2 values hidden

""")

    def test_ex1_goal4_step6_extendString(self):
        s, cardnum, person = self.initialise()
        t31 = s.get([cardnum['val_id']])
        t32 = t31.infer(['val_id'], cardnum['cardnum'])
        t33 = t32.infer(['cardnum'], person['cardnum'], with_name="person.cardnum")
        t34 = t33.infer(['person.cardnum'], person['person'])
        t35 = t34.hide('person.cardnum').hide('cardnum')
        t36 = t35.show('person.cardnum').show('cardnum')
        t37 = t36.extend("person", "Bob", "person_fillna")
        self.assertExpectedInline(str(t37), """\
[val_id || cardnum person.cardnum person person_fillna]
        cardnum  person.cardnum person person_fillna
val_id                                              
4.0      1111.0          1111.0  Steve         Steve
3.0      1410.0          1410.0    Tom           Tom
2.0      2354.0          2354.0  Steve         Steve
5.0      2354.0          2354.0  Steve         Steve
1.0      5172.0          5172.0    NaN           Bob
8.0      4412.0          4412.0    NaN           Bob
4 values hidden

""")

    def test_ex1_goal5_step1_getAndCompose(self):
        s, cardnum, person = self.initialise()
        t41 = (s.get([person['cardnum']]).infer(['cardnum'], person['person']))
        t42 = t41.compose([cardnum['val_id']], 'cardnum')
        self.assertExpectedInline(str(t42), """\
[val_id || person]
       person
val_id       
2.0     Steve
5.0     Steve
3.0       Tom
4.0     Steve
2 keys hidden
2 values hidden

""")

    def test_ex1_goal5_step2_setKey(self):
        s, cardnum, person = self.initialise()
        t41 = (s.get([person['cardnum']]).infer(['cardnum'], person['person']))
        t42 = t41.compose([cardnum['val_id']], 'cardnum')
        t43 = t42.set_key(["person"])
        self.assertExpectedInline(str(t43), """\
[person || val_id]
                 val_id
person                 
Steve   [2.0, 5.0, 4.0]
Tom               [3.0]
3 keys hidden

""")