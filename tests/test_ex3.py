import expecttest
import numpy as np
import pandas as pd

from schema import Schema, SchemaNode

s = Schema()

cardnum = pd.read_csv("./csv/bonuses/cardnum.csv").set_index("val_id")
tstart = pd.read_csv("./csv/bonuses/tstart.csv").set_index("val_id")
bonus = pd.read_csv("./csv/bonuses/bonus.csv").set_index(["val_id", "cardnum"])

s.insert_dataframe(cardnum, "cardnum")
s.insert_dataframe(tstart, "tstart")
s.insert_dataframe(bonus, "bonus")

c_cardnum = SchemaNode("cardnum", cluster="cardnum")
b_cardnum = SchemaNode("cardnum", cluster="bonus")
c_val_id = SchemaNode("val_id", cluster="cardnum")
t_val_id = SchemaNode("val_id", cluster="tstart")
b_val_id = SchemaNode("val_id", cluster="bonus")

s.blend(c_val_id, t_val_id, under="Val_id")
s.blend(c_val_id, b_val_id)
s.blend(c_cardnum, b_cardnum, under="Cardnum")

# ========================================================================
# ========================================================================

class TestEx3(expecttest.TestCase):
    def test_ex3(self):
        # SCHEMA:
        # cardnum <--- val_id ---> t_start
        # val_id, cardnum ---> bonus

        # GOAL 1: I want to know, for each val_id cardnum pair, what the bonus is, and what the t_start is

        # Get every Val_id, Cardnum pair
        t1 = s.get(["Val_id", "Cardnum"])
        print(t1)
        # [Val_id Cardnum || ]
        #  1      5172
        #  1      1111
        #  1      1410
        #  1      2354
        #  1      6440
        #  1      5467
        #  2      5172
        #  2      1111
        #  2      1410
        #  2      2354
        #  2      6440
        #  2      5467
        #  ...
        #  8      1111
        #  8      1410
        #  8      2354
        #  8      6440
        #  8      5467

        # # # From val_id, cardnum, I can tell you the bonus
        # # # This will trim the key set, since values populate keys
        t2 = t1.infer(["Val_id", "Cardnum"], "bonus.bonus")
        print(t2)
        # # # [Val_id Cardnum || bonus.bonus]
        # # #  1      5172    || 4
        # # #  1      1410    || 12
        # # #  2      1111    || 5
        # # #  2      6440    || 7
        # # #  3      1111    || 1
        # # #  5      1410    || 2
        # #
        # # # But we know for each val_id, we can infer a tstart
        t3 = t2.infer(["Val_id"], "tstart.tstart")
        print(t3)
        # # # [Val_id Cardnum || bonus.bonus  tstart.tstart]
        # # #  1      5172    || 4            2023-01-01 09:50:00
        # # #  1      1111    || NA           2023-01-01 09:50:00
        # # #  1      1410    || 12           2023-01-01 09:50:00
        # # #  1      2354    || NA           2023-01-01 09:50:00
        # # #  1      6440    || NA           2023-01-01 09:50:00
        # # #  1      5467    || NA           2023-01-01 09:50:00
        # # #  ...
        # #
        # # # I'm only interested in rows where the bonus actually exists.
        t4 = t3.filter(t3["bonus.bonus"].isnotnull())
        print(t4)
        # # # [Val_id Cardnum || bonus.bonus  tstart.tstart]
        # # #  1      5172    || 4            2023-01-01 09:50:00
        # # #  1      1410    || 12           2023-01-01 09:50:00
        # # #  2      1111    || 5            2023-01-01 11:10:00
        # # #  2      6440    || 7            2023-01-01 11:10:00
        # # #  3      1111    || 1            2023-01-01 15:32:00
        # # #  5      1410    || 2            2023-01-01 20:11:00
        # #
        # # # Did anyone actually get a bonus?
        t5 = t4.infer(["Val_id"], "cardnum.cardnum").filter(t4["bonus.bonus"].isnotnull())
        print(t5)
        # # # Values populate keys. Since we use the same values, we will end up with the same keys.
        # # # [Val_id Cardnum || bonus.bonus  tstart.tstart            cardnum.cardnum]
        # # #  1      5172    || 4            2023-01-01 09:50:00      5172
        # # #  1      1410    || 12           2023-01-01 09:50:00      5172
        # # #  2      1111    || 5            2023-01-01 11:10:00      2354
        # # #  2      6440    || 7            2023-01-01 11:10:00      2354
        # # #  3      1111    || 1            2023-01-01 15:32:00      1410
        # # #  5      1410    || 2            2023-01-01 20:11:00      2354
        # #
        t6 = t5.filter(t5["Cardnum"] == t5["cardnum.cardnum"])
        print(t6)
        # # [Val_id Cardnum || bonus.bonus  tstart.tstart            cardnum.cardnum]
        # #  1      5172    || 4            2023-01-01 09:50:00      5172

    def test_ex3_goal2(self):
        # GOAL 2: [val_id || cardnum tstart bonus]
        t11 = s.get(["Val_id"])
        print(t11)
        # [Val_id || ]
        #  1
        #  2
        #  3
        #  4
        #  5
        #  6
        #  7
        #  8

        t12 = (t11.infer(["Val_id"], "cardnum.cardnum")
               .infer(["Val_id"], "tstart.tstart")
               .infer(["Val_id"], "bonus.bonus"))
        print(t12)
        # [Val_id || tstart.tstart         cardnum.cardnum   bonus.bonus]
        #  1      || 2023-01-01 09:50:00   5172              [4, 12]
        #  2      || 2023-01-01 11:10:00   2354              [5, 7]
        #  3      || 2023-01-01 15:32:00   1410              [1]
        #  4      || 2023-01-01 15:34:00   1111              []
        #  5      || 2023-01-01 20:11:00   2354              [2]
        #  6      || 2023-01-01 21:17:00   NA                []
        #  7      || 2023-01-02 05:34:00   NA                []
        #  8      || NA                    4412              []

        # bonus.cardnum is a hidden key for bonus.bonus. Let's show it.
        # this hides rows for which bonus.bonus is [], since it implies bonus.cardnum is NA
        # this is the same as t5!
        # Does this mean that hiding a key could introduce new rows - specifically rows where
        # the hidden key is NA? Yes.
        t13 = t12.show("bonus.cardnum")
        print(t13)
        # [Val_id bonus.cardnum || tstart.tstart         cardnum.cardnum   bonus.bonus]
        #  1      5172          || 2023-01-01 09:50:00   5172              4
        #  1      1410          || 2023-01-01 09:50:00   5172              12
        #  2      1111          || 2023-01-01 11:10:00   2354              5
        #  2      6440          || 2023-01-01 11:10:00   2354              7
        #  3      1111          || 2023-01-01 15:32:00   1410              1
        #  5      1410          || 2023-01-01 20:11:00   2354              2
