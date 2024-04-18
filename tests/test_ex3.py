import expecttest
import numpy as np
import pandas as pd

from schema.schema import Schema
from schema.node import SchemaNode


class TestEx3(expecttest.TestCase):

    def initialise(self):
        s = Schema()

        cardnum_df = pd.read_csv("./csv/bonuses/cardnum.csv").set_index("val_id")
        tstart_df = pd.read_csv("./csv/bonuses/tstart.csv").set_index("val_id")
        bonus_df = pd.read_csv("./csv/bonuses/bonus.csv").set_index(
            ["val_id", "cardnum"]
        )

        cardnum = s.insert_dataframe(cardnum_df)
        tstart = s.insert_dataframe(tstart_df)
        bonus = s.insert_dataframe(bonus_df)

        c_cardnum = cardnum["cardnum"]
        b_cardnum = bonus["cardnum"]
        c_val_id = cardnum["val_id"]
        t_val_id = tstart["val_id"]
        b_val_id = bonus["val_id"]

        Cardnum = s.create_class("Cardnum")
        Val_id = s.create_class("Val_id")

        s.blend(c_val_id, t_val_id, Val_id)
        s.blend(c_val_id, b_val_id)
        s.blend(c_cardnum, b_cardnum, Cardnum)
        return s, bonus, cardnum, tstart, Cardnum, Val_id

        # SCHEMA:
        # cardnum <--- val_id ---> t_start
        # val_id, cardnum ---> bonus

    # GOAL 1: I want to know, for each val_id cardnum pair, what the bonus is, and what the t_start is

    def test_ex3_goal1_step1_get(self):
        # Get every Val_id, Cardnum pair
        s, bonus, cardnum, tstart, Cardnum, Val_id = self.initialise()
        t1 = s.get(Val_id=Val_id, Cardnum=Cardnum).sort(["Val_id", "Cardnum"])
        self.maxDiff = None
        self.assertExpectedInline(
            str(t1),
            """\
[Val_id Cardnum || ]
Empty DataFrame
Columns: []
Index: []
48 keys hidden

""",
        )
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

    def test_ex3_goal1_step2_infer(self):
        # From val_id, cardnum, I can tell you the bonus
        # This will trim the key set, since values populate keys
        s, bonus, cardnum, tstart, Cardnum, Val_id = self.initialise()
        t1 = s.get(Val_id=Val_id, Cardnum=Cardnum)
        t2 = t1.infer(["Val_id", "Cardnum"], bonus["bonus"])
        self.assertExpectedInline(
            str(t2),
            """\
[Val_id Cardnum || bonus]
                bonus
Val_id Cardnum       
1      5172       4.0
2      1111       5.0
3      1111       1.0
1      1410      12.0
2      6440       7.0
5      1410       2.0
42 keys hidden

""",
        )
        # [Val_id Cardnum || bonus.bonus]
        #  1      5172    || 4
        #  1      1410    || 12
        #  2      1111    || 5
        #  2      6440    || 7
        #  3      1111    || 1
        #  5      1410    || 2

    def test_ex3_goal1_step3_infer(self):
        # But we know for each val_id, we can infer a tstart
        s, bonus, cardnum, tstart, Cardnum, Val_id = self.initialise()
        t1 = s.get(Val_id=Val_id, Cardnum=Cardnum)
        t2 = t1.infer(["Val_id", "Cardnum"], bonus["bonus"])
        t3 = t2.infer(["Val_id"], tstart["tstart"]).sort(["Val_id", "Cardnum"])
        self.maxDiff = None
        self.assertExpectedInline(
            str(t3),
            """\
[Val_id Cardnum || bonus tstart]
                bonus               tstart
Val_id Cardnum                            
1      1111       NaN  2023-01-01 09:50:00
       1410      12.0  2023-01-01 09:50:00
       2354       NaN  2023-01-01 09:50:00
       4412       NaN  2023-01-01 09:50:00
       5172       4.0  2023-01-01 09:50:00
       6440       NaN  2023-01-01 09:50:00
2      1111       5.0  2023-01-01 11:10:00
       1410       NaN  2023-01-01 11:10:00
       2354       NaN  2023-01-01 11:10:00
       4412       NaN  2023-01-01 11:10:00
       5172       NaN  2023-01-01 11:10:00
       6440       7.0  2023-01-01 11:10:00
3      1111       1.0  2023-01-01 15:32:00
       1410       NaN  2023-01-01 15:32:00
       2354       NaN  2023-01-01 15:32:00
       4412       NaN  2023-01-01 15:32:00
       5172       NaN  2023-01-01 15:32:00
       6440       NaN  2023-01-01 15:32:00
4      1111       NaN  2023-01-01 15:34:00
       1410       NaN  2023-01-01 15:34:00
       2354       NaN  2023-01-01 15:34:00
       4412       NaN  2023-01-01 15:34:00
       5172       NaN  2023-01-01 15:34:00
       6440       NaN  2023-01-01 15:34:00
5      1111       NaN  2023-01-01 20:11:00
       1410       2.0  2023-01-01 20:11:00
       2354       NaN  2023-01-01 20:11:00
       4412       NaN  2023-01-01 20:11:00
       5172       NaN  2023-01-01 20:11:00
       6440       NaN  2023-01-01 20:11:00
6      1111       NaN  2023-01-01 21:17:00
       1410       NaN  2023-01-01 21:17:00
       2354       NaN  2023-01-01 21:17:00
       4412       NaN  2023-01-01 21:17:00
       5172       NaN  2023-01-01 21:17:00
       6440       NaN  2023-01-01 21:17:00
7      1111       NaN  2023-01-02 05:34:00
       1410       NaN  2023-01-02 05:34:00
       2354       NaN  2023-01-02 05:34:00
       4412       NaN  2023-01-02 05:34:00
       5172       NaN  2023-01-02 05:34:00
       6440       NaN  2023-01-02 05:34:00
6 keys hidden

""",
        )
        # [Val_id Cardnum || bonus.bonus  tstart.tstart]
        #  1      5172    || 4            2023-01-01 09:50:00
        #  1      1111    || NA           2023-01-01 09:50:00
        #  1      1410    || 12           2023-01-01 09:50:00
        #  1      2354    || NA           2023-01-01 09:50:00
        #  1      6440    || NA           2023-01-01 09:50:00
        #  1      5467    || NA           2023-01-01 09:50:00
        #  ...

    def test_ex3_goal1_step4_filter(self):
        # I'm only interested in rows where the bonus actually exists.
        s, bonus, cardnum, tstart, Cardnum, Val_id = self.initialise()
        t1 = s.get(Val_id=Val_id, Cardnum=Cardnum)
        t2 = t1.infer(["Val_id", "Cardnum"], bonus["bonus"])
        t3 = t2.infer(["Val_id"], tstart["tstart"])
        t4 = t3.filter(t3["bonus"])
        self.assertExpectedInline(
            str(t4),
            """\
[Val_id Cardnum || bonus tstart]
                bonus               tstart
Val_id Cardnum                            
1      5172       4.0  2023-01-01 09:50:00
2      1111       5.0  2023-01-01 11:10:00
3      1111       1.0  2023-01-01 15:32:00
1      1410      12.0  2023-01-01 09:50:00
2      6440       7.0  2023-01-01 11:10:00
5      1410       2.0  2023-01-01 20:11:00
42 keys hidden

""",
        )
        # [Val_id Cardnum || bonus.bonus  tstart.tstart]
        #  1      5172    || 4            2023-01-01 09:50:00
        #  1      1410    || 12           2023-01-01 09:50:00
        #  2      1111    || 5            2023-01-01 11:10:00
        #  2      6440    || 7            2023-01-01 11:10:00
        #  3      1111    || 1            2023-01-01 15:32:00
        #  5      1410    || 2            2023-01-01 20:11:00

    def test_ex3_goal1_step5_infer(self):
        # Did anyone actually get a bonus?
        s, bonus, cardnum, tstart, Cardnum, Val_id = self.initialise()
        t1 = s.get(Val_id=Val_id, Cardnum=Cardnum)
        t2 = t1.infer(["Val_id", "Cardnum"], bonus["bonus"])
        t3 = t2.infer(["Val_id"], tstart["tstart"])
        t4 = t3.infer(["Val_id"], cardnum["cardnum"])
        t5 = t4.deduce(t4["bonus"] / 100 * t4["cardnum"], "return").filter("return")
        self.maxDiff = None
        self.assertExpectedInline(
            str(t5),
            """\
[Val_id Cardnum || bonus tstart cardnum return]
                bonus               tstart  cardnum  return
Val_id Cardnum                                             
1      5172       4.0  2023-01-01 09:50:00   5172.0  206.88
2      1111       5.0  2023-01-01 11:10:00   2354.0  117.70
3      1111       1.0  2023-01-01 15:32:00   1410.0   14.10
1      1410      12.0  2023-01-01 09:50:00   5172.0  620.64
2      6440       7.0  2023-01-01 11:10:00   2354.0  164.78
5      1410       2.0  2023-01-01 20:11:00   2354.0   47.08
42 keys hidden

""",
        )
        # Values populate keys. Since we use the same values, we will end up with the same keys.
        # [Val_id Cardnum || bonus.bonus  tstart.tstart            cardnum.cardnum]
        #  1      5172    || 4            2023-01-01 09:50:00      5172
        #  1      1410    || 12           2023-01-01 09:50:00      5172
        #  2      1111    || 5            2023-01-01 11:10:00      2354
        #  2      6440    || 7            2023-01-01 11:10:00      2354
        #  3      1111    || 1            2023-01-01 15:32:00      1410
        #  5      1410    || 2            2023-01-01 20:11:00      2354

    def test_ex3_goal1_step6_filter(self):
        # Did anyone actually get a bonus?
        s, bonus, cardnum, tstart, Cardnum, Val_id = self.initialise()
        t1 = s.get(Val_id=Val_id, Cardnum=Cardnum)
        t2 = t1.infer(["Val_id", "Cardnum"], bonus["bonus"])
        t3 = t2.infer(["Val_id"], tstart["tstart"])
        t4 = t3.infer(["Val_id"], cardnum["cardnum"])
        t5 = t4.mutate(
            Cardnum_filtered=t4["Cardnum"].mask(
                (t4["Cardnum"] == t4["cardnum"]) & t4["bonus"].isnotnull()
            )
        )
        t6 = t5.filter(t5["Cardnum_filtered"])
        self.maxDiff = None
        self.assertExpectedInline(
            str(t6),
            """\
[Val_id Cardnum || bonus tstart cardnum Cardnum_filtered]
                bonus               tstart  cardnum Cardnum_filtered
Val_id Cardnum                                                      
1      5172       4.0  2023-01-01 09:50:00   5172.0             5172
47 keys hidden

""",
        )
        # # [Val_id Cardnum || bonus.bonus  tstart.tstart            cardnum.cardnum]
        # #  1      5172    || 4            2023-01-01 09:50:00      5172

    # GOAL 2: [val_id || cardnum tstart bonus]
    def test_ex3_goal2_step1_get(self):
        s, bonus, cardnum, tstart, Cardnum, Val_id = self.initialise()
        t11 = s.get(Val_id=Val_id)
        self.assertExpectedInline(
            str(t11),
            """\
[Val_id || ]
Empty DataFrame
Columns: []
Index: []
8 keys hidden

""",
        )
        # [Val_id || ]
        #  1
        #  2
        #  3
        #  4
        #  5
        #  6
        #  7
        #  8

    def test_ex3_goal2_step2_inferenceChain(self):
        s, bonus, cardnum, tstart, Cardnum, Val_id = self.initialise()
        t11 = s.get(Val_id=Val_id)
        t12 = (
            t11.infer(["Val_id"], cardnum["cardnum"])
            .infer(["Val_id"], tstart["tstart"])
            .infer(["Val_id"], bonus["bonus"])
        ).sort("Val_id")
        self.assertExpectedInline(
            str(t12),
            """\
[Val_id || cardnum tstart bonus]
        cardnum               tstart        bonus
Val_id                                           
1        5172.0  2023-01-01 09:50:00  [4.0, 12.0]
2        2354.0  2023-01-01 11:10:00   [5.0, 7.0]
3        1410.0  2023-01-01 15:32:00        [1.0]
4        1111.0  2023-01-01 15:34:00           []
5        2354.0  2023-01-01 20:11:00        [2.0]
6           NaN  2023-01-01 21:17:00           []
7           NaN  2023-01-02 05:34:00           []
8        4412.0                  NaN           []

""",
        )
        # [Val_id || tstart.tstart         cardnum.cardnum   bonus.bonus]
        #  1      || 2023-01-01 09:50:00   5172              [4, 12]
        #  2      || 2023-01-01 11:10:00   2354              [5, 7]
        #  3      || 2023-01-01 15:32:00   1410              [1]
        #  4      || 2023-01-01 15:34:00   1111              []
        #  5      || 2023-01-01 20:11:00   2354              [2]
        #  6      || 2023-01-01 21:17:00   NA                []
        #  7      || 2023-01-02 05:34:00   NA                []
        #  8      || NA                    4412              []

    def test_ex3_goal2_step3_assignAndAggregate(self):
        s, bonus, cardnum, tstart, Cardnum, Val_id = self.initialise()
        t11 = s.get(Val_id=Val_id)
        t12 = (
            t11.infer(["Val_id"], cardnum["cardnum"])
            .infer(["Val_id"], tstart["tstart"])
            .infer(["Val_id"], bonus["bonus"])
        )
        t13 = t12.deduce(
            (t12["bonus"].sum() / t12["bonus"].count() / 100) * t12["cardnum"],
            "returns",
        ).sort("Val_id")
        print(t13.intermediate_representation)
        self.maxDiff = None
        print(t13["returns"].get_hidden_keys())
        print(t13.derivation)
        self.assertExpectedInline(
            str(t13),
            """\
[Val_id || cardnum tstart bonus returns]
        cardnum               tstart        bonus  returns
Val_id                                                    
1        5172.0  2023-01-01 09:50:00  [4.0, 12.0]   413.76
2        2354.0  2023-01-01 11:10:00   [5.0, 7.0]   141.24
3        1410.0  2023-01-01 15:32:00        [1.0]    14.10
4        1111.0  2023-01-01 15:34:00           []      NaN
5        2354.0  2023-01-01 20:11:00        [2.0]    47.08
6           NaN  2023-01-01 21:17:00           []      NaN
7           NaN  2023-01-02 05:34:00           []      NaN
8        4412.0                  NaN           []      NaN

""",
        )

    def test_ex3_goal2_step3_showingAHiddenKey(self):
        # bonus.cardnum is a hidden key for bonus.bonus. Let's show it.
        # this hides rows for which bonus.bonus is [], since it implies bonus.cardnum is NA
        # this is the same as t5!
        # Does this mean that hiding a key could introduce new rows - specifically rows where
        # the hidden key is NA? Yes.
        s, bonus, cardnum, tstart, Cardnum, Val_id = self.initialise()
        t11 = s.get(Val_id=Val_id)
        t12 = (
            t11.infer(["Val_id"], cardnum["cardnum"])
            .infer(["Val_id"], tstart["tstart"])
            .infer(["Val_id"], bonus["bonus"])
        )
        t14 = t12.show("cardnum_1").filter("bonus")
        print("T14 DERIVATION")
        print(t14.derivation)
        self.maxDiff = None
        self.assertExpectedInline(
            str(t14),
            """\
[Val_id cardnum_1 || cardnum tstart bonus]
                  cardnum               tstart  bonus
Val_id cardnum_1                                     
1      5172        5172.0  2023-01-01 09:50:00    4.0
2      1111        2354.0  2023-01-01 11:10:00    5.0
3      1111        1410.0  2023-01-01 15:32:00    1.0
1      1410        5172.0  2023-01-01 09:50:00   12.0
2      6440        2354.0  2023-01-01 11:10:00    7.0
5      1410        2354.0  2023-01-01 20:11:00    2.0
26 keys hidden

""",
        )
        # [Val_id bonus.cardnum || tstart.tstart         cardnum.cardnum   bonus.bonus]
        #  1      5172          || 2023-01-01 09:50:00   5172              4
        #  1      1410          || 2023-01-01 09:50:00   5172              12
        #  2      1111          || 2023-01-01 11:10:00   2354              5
        #  2      6440          || 2023-01-01 11:10:00   2354              7
        #  3      1111          || 2023-01-01 15:32:00   1410              1
        #  5      1410          || 2023-01-01 20:11:00   2354              2
