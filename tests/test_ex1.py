import expecttest
import pandas as pd

from schema import Schema, SchemaNode

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
c_val_id = SchemaNode("val_id", cluster="cardnum")
p_cardnum = SchemaNode("cardnum", cluster="person")
p_person = SchemaNode("person", cluster="person")

s.blend(c_cardnum, p_cardnum, under="Cardnum")


class TestEx1(expecttest.TestCase):

    # SCHEMA:
    # val_id ---> cardnum ---> person

    # GOAL 1: Find all people who made trips
    # Get the cardnum for each trip, use it to infer who made the trip,
    # and then restrict it to valid trips
    def test_ex1_goal1_step1(self):

        t1 = s.get(["cardnum.cardnum"])
        print(t1)
        # Get every cardnum in the cardnum csv
        # [cardnum.cardnum || ]
        #  5172
        #  2354
        #  1410
        #  1111
        #  4412

    def test_ex1_goal1_step2(self):
        t1 = s.get(["cardnum.cardnum"])
        t2 = t1.infer(["cardnum.cardnum"], "person.person")
        print(t2)
        # Use this cardnum to infer who made the trip
        # Dropped keys! Values populate keys.
        # [cardnum.cardnum || person.person]
        #  2354            || Steve
        #  1410            || Tom
        #  1111            || Steve

    def test_ex1_goal1_step3(self):
        t1 = s.get(["cardnum.cardnum"])
        t2 = t1.infer(["cardnum.cardnum"], "person.cardnum")
        t3 = t2.infer(["person.cardnum"], "person.person")
        # t3 = t2.compose(["cardnum.val_id"], "cardnum.cardnum")
        print(t3.hide("person.person"))
        # # Hey, I know how to get the val_id(s) given the cardnum
        # # [cardnum.val_id || person.person]
        # # 2               || Steve
        # # 5               || Steve
        # # 3               || Tom
        # # 4               || Steve


        # # But what if, as a byproduct, I want to know all people who have cardnums?
        # # GOAL 2: Same as goal 1, but include people who didn't make trips
        # # Get the cardnum for each PERSON, use it to infer who made the trip,
        # # and then restrict it to valid trips

    def test_ex1_goal2_step1(self):
        t11 = s.get(["person.cardnum"])
        print(t11)
        # # Get me every value of cardnum in person
        # # [person.cardnum || ]
        # #  1111
        # #  1410
        # #  2354
        # #  6440
        # #  5467

        t12 = t11.infer(["person.cardnum"], "person.person")
        print(t12)
        # # From the value of cardnum, tell me who the card belongs to
        # # [person.cardnum  || person.person]
        # #  1111            || Steve
        # #  1410            || Tom
        # #  2354            || Steve
        # #  6440            || Harry
        # #  5467            || Dick
        #
        # # If that's all I want, I can stop here! But if I want to know which people
        # # did NOT make trips, I can simply do
        #
        t13 = t12.compose(["cardnum.val_id"], "person.cardnum")
        print(t13)
        # # I have a mapping from val_id to cardnum
        # # [cardnum.val_id || person.person]
        # # 2               || Steve
        # # 5               || Steve
        # # 3               || Tom
        # # 4               || Steve

    def test_ex1_goal3_step1(self):
        t21 = s.get(["cardnum.cardnum"])
        t22 = t21.infer(["cardnum.cardnum"], "cardnum.val_id")
        print(t22)
        # Get every cardnum in the cardnum csv
        # [cardnum.cardnum || ]
        #  5172
        #  2354
        #  1410
        #  1111
        #  4412

    def test_ex1_goal4(self):
        table1 = s.get(['cardnum.val_id'])
        print(table1)
        table2 = table1.infer(['cardnum.val_id'], 'cardnum.cardnum')
        print(table2)
        table3 = table2.infer(['cardnum.cardnum'], 'person.cardnum')
        print(table3)
        table4 = table3.infer(['person.cardnum'], 'person.person')
        print(table4)
        table = table4.hide('person.cardnum').hide('cardnum.cardnum')
        print(table)


