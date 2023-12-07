import expecttest
import pandas as pd

from schema import Schema, SchemaNode


class TestEx1(expecttest.TestCase):
    def test_ex1(self):
        s = Schema()

        # Load up the CSVs
        cardnum = pd.read_csv("./csv/bonuses/cardnum.csv").set_index("val_id")
        person = pd.read_csv("./csv/bonuses/person.csv").set_index("person")

        # Insert them into the data frames
        s.insert_dataframe(cardnum, "cardnum")
        s.insert_dataframe(person, "person")

        # Get handles on the nodes now in the schema - may be slightly ugly but
        # we can slap on a load of syntactic sugar
        c_cardnum = SchemaNode("cardnum", cluster="cardnum")
        c_val_id = SchemaNode("val_id", cluster="cardnum")
        p_cardnum = SchemaNode("cardnum", cluster="person")
        p_person = SchemaNode("person", cluster="person")

        s.blend(c_cardnum, p_cardnum, as="Cardnum")

        # ========================================================================
        # ========================================================================
        # (I'll use this to denote the boundary between schema creation and table querying,
        # so you can skip over the schema creation)
        # Do note that I've created .csv files and you can find their paths above, though.

        # SCHEMA:
        # val_id ---> cardnum ---> person

        # GOAL 1: Find all people who made trips

        # Get the cardnum for each trip, use it to infer who made the trip,
        # and then restrict it to valid trips

        t1 = s.get([c_cardnum])
        # Get every cardnum in the cardnum csv
        # [cardnum.cardnum || ]
        #  5172
        #  2354
        #  1410
        #  1111
        #  4412

        t2 = t1.infer(["cardnum.cardnum"], p_person)
        # Use this cardnum to infer who made the trip
        # Dropped keys! Values populate keys.
        # [cardnum.cardnum || person.person]
        #  2354            || Steve
        #  1410            || Tom
        #  1111            || Steve


        t3 = t2.compose(c_val_id, ["cardnum.cardnum"])
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

        t11 = s.get([p_cardnum])
        # Get me every value of cardnum in person
        # [person.cardnum || ]
        #  1111
        #  1410
        #  2354
        #  6440
        #  5467

        t12 = t11.infer(["Cardnum"], p_person)
        # From the value of cardnum, tell me who the card belongs to
        # [person.cardnum  || person.person]
        #  1111            || Steve
        #  1410            || Tom
        #  2354            || Steve
        #  6440            || Harry
        #  5467            || Dick

        # If that's all I want, I can stop here! But if I want to know which people
        # did NOT make trips, I can simply do

        t13 = t12.compose(c_val_id, ["person.cardnum"])
        # I have a mapping from val_id to cardnum
        # [cardnum.val_id || person.person]
        # 2               || Steve
        # 5               || Steve
        # 3               || Tom
        # 4               || Steve

