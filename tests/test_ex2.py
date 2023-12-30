import expecttest
import pandas as pd

from schema import Schema, SchemaNode


class TestEx2(expecttest.TestCase):
    def initialise(self):
        # SCHEMA:
        # cardnum <--- val_id ---> t_start
        s = Schema()
        cardnum_df = pd.read_csv("./csv/bonuses/cardnum.csv").set_index("val_id")
        tstart_df = pd.read_csv("./csv/bonuses/tstart.csv").set_index("val_id")
        cardnum = s.insert_dataframe(cardnum_df)
        tstart = s.insert_dataframe(tstart_df)

        c_val_id = cardnum["val_id"]
        t_val_id = tstart["val_id"]

        Val_id = s.create_class("Val_id")

        s.blend(c_val_id, t_val_id, under=Val_id)
        return s, cardnum, tstart

# GOAL: For each trip (val_id), tell me when the trip started (t_start)
# Only two obvious ways to do it

    def test_ex2_goal1_step1_get(self):
        s, cardnum, tstart = self.initialise()
        t1 = s.get([tstart["val_id"]])
        self.assertExpectedInline(str(t1), """\
[val_id || ]
Empty DataFrame
Columns: []
Index: [1, 2, 3, 4, 5, 6, 7]

""")
        # Get every val_id in the tstart csv
        # [tstart.val_id || ]
        #  1
        #  2
        #  3
        #  4
        #  5
        #  6
        #  7

    def test_ex2_goal1_step2_infer(self):
        s, cardnum, tstart = self.initialise()
        t1 = s.get([tstart["val_id"]])
        t2 = t1.infer(["tstart.val_id"], "tstart.tstart")
        self.assertExpectedInline(str(t2), """\
[tstart.val_id || tstart.tstart]
                     tstart.tstart
tstart.val_id                     
1              2023-01-01 09:50:00
2              2023-01-01 11:10:00
3              2023-01-01 15:32:00
4              2023-01-01 15:34:00
5              2023-01-01 20:11:00
6              2023-01-01 21:17:00
7              2023-01-02 05:34:00

""")
        # [tstart.val_id || tstart.tstart]
        #  1             || 2023-01-01 09:50:00
        #  2             || 2023-01-01 11:10:00
        #  3             || 2023-01-01 15:32:00
        #  4             || 2023-01-01 15:34:00
        #  5             || 2023-01-01 20:11:00
        #  6             || 2023-01-01 21:17:00
        #  7             || 2023-01-02 05:34:00

    def test_ex2_goal2_step1_get(self):
        s = self.initialise()
        t11 = s.get(["Val_id"])
        self.assertExpectedInline(str(t11), """\
[Val_id || ]
Empty DataFrame
Columns: []
Index: [1, 2, 3, 4, 5, 8, 6, 7]

""")
        # Get every possible val_id. Note that there is val_id 8, but we don't
        # have a timestamp for that.
        # [Val_id || ]
        #  1
        #  2
        #  3
        #  4
        #  5
        #  6
        #  7
        #  8

    def test_ex2_goal2_step2_infer(self):
        s = self.initialise()
        t11 = s.get(["Val_id"])
        t12 = t11.infer(["Val_id"], "tstart.tstart")
        self.assertExpectedInline(str(t12), """\
[Val_id || tstart.tstart]
              tstart.tstart
Val_id                     
1       2023-01-01 09:50:00
2       2023-01-01 11:10:00
3       2023-01-01 15:32:00
4       2023-01-01 15:34:00
5       2023-01-01 20:11:00
6       2023-01-01 21:17:00
7       2023-01-02 05:34:00
1 keys hidden

""")
        # Values populate keys. Since we use the same values, we will end up with the same keys.
        # [Val_id || tstart.tstart]
        #  1      || 2023-01-01 09:50:00
        #  2      || 2023-01-01 11:10:00
        #  3      || 2023-01-01 15:32:00
        #  4      || 2023-01-01 15:34:00
        #  5      || 2023-01-01 20:11:00
        #  6      || 2023-01-01 21:17:00
        #  7      || 2023-01-02 05:34:00

    def test_ex2_goal3_step1_inferAgain(self):
        s = self.initialise()
        t11 = s.get(["Val_id"])
        t12 = t11.infer(["Val_id"], "tstart.tstart")
        # Hey, I also want the cardnum, not just the val_id
        # First two steps are the same as either Way 1 or Way 2, but then you can do
        t23 = t12.infer(["Val_id"], "cardnum.cardnum")
        self.assertExpectedInline(str(t23), """\
[Val_id || tstart.tstart cardnum.cardnum]
              tstart.tstart  cardnum.cardnum
Val_id                                      
1       2023-01-01 09:50:00           5172.0
2       2023-01-01 11:10:00           2354.0
3       2023-01-01 15:32:00           1410.0
4       2023-01-01 15:34:00           1111.0
5       2023-01-01 20:11:00           2354.0
8                       NaN           4412.0
6       2023-01-01 21:17:00              NaN
7       2023-01-02 05:34:00              NaN

""")
        # [Val_id || tstart.tstart            cardnum.cardnum]
        #  1      || 2023-01-01 09:50:00      5172
        #  2      || 2023-01-01 11:10:00      2354
        #  3      || 2023-01-01 15:32:00      1410
        #  4      || 2023-01-01 15:34:00      1111
        #  5      || 2023-01-01 20:11:00      2354
        #  6      || 2023-01-01 21:17:00      NA
        #  7      || 2023-01-02 05:34:00      NA
        #  8      || NA                       4412

        # Note the presence of NA values! Values populate keys:
        # A key exists as long as ONE value is not NA.

        # STRESS TEST
    def test_ex2_goal4_step1_getCrossProduct(self):
        # What if the user accidentally makes cardnum a key, so gets the cross product?
        s = self.initialise()
        t31 = s.get(["Val_id", "cardnum.cardnum"])
        self.assertExpectedInline(str(t31), """\
[Val_id cardnum.cardnum || ]
Empty DataFrame
Columns: []
Index: [(1, 5172), (1, 2354), (1, 1410), (1, 1111), (1, 4412), (2, 5172), (2, 2354), (2, 1410), (2, 1111), (2, 4412), (3, 5172), (3, 2354), (3, 1410), (3, 1111), (3, 4412), (4, 5172), (4, 2354), (4, 1410), (4, 1111), (4, 4412), (5, 5172), (5, 2354), (5, 1410), (5, 1111), (5, 4412), (8, 5172), (8, 2354), (8, 1410), (8, 1111), (8, 4412), (6, 5172), (6, 2354), (6, 1410), (6, 1111), (6, 4412), (7, 5172), (7, 2354), (7, 1410), (7, 1111), (7, 4412)]

""")
        # [Val_id cardnum.cardnum || ]
        #  1      5172
        #  1      2354
        #  1      1410
        #  1      1111
        #  2      2354
        # ....
        #  8      5172
        #  8      2354
        #  8      1410
        #  8      1111

    def test_ex2_goal4_step2_infer(self):
        # Oh... But values populate keys, maybe inference will help a bit?
        # Helps a bit, but not much
        # cardnum is a weak key for tstart, but I think it will be too surprising to suddenly drop it.
        # what if the user is keeping it around for something else?
        s = self.initialise()
        t31 = s.get(["Val_id", "cardnum.cardnum"])
        t32 = t31.infer(["Val_id"], "tstart.tstart")
        self.assertExpectedInline(str(t32), """\
[Val_id cardnum.cardnum || tstart.tstart]
                              tstart.tstart
Val_id cardnum.cardnum                     
1      5172             2023-01-01 09:50:00
       2354             2023-01-01 09:50:00
       1410             2023-01-01 09:50:00
       1111             2023-01-01 09:50:00
       4412             2023-01-01 09:50:00
2      5172             2023-01-01 11:10:00
       2354             2023-01-01 11:10:00
       1410             2023-01-01 11:10:00
       1111             2023-01-01 11:10:00
       4412             2023-01-01 11:10:00
3      5172             2023-01-01 15:32:00
       2354             2023-01-01 15:32:00
       1410             2023-01-01 15:32:00
       1111             2023-01-01 15:32:00
       4412             2023-01-01 15:32:00
4      5172             2023-01-01 15:34:00
       2354             2023-01-01 15:34:00
       1410             2023-01-01 15:34:00
       1111             2023-01-01 15:34:00
       4412             2023-01-01 15:34:00
5      5172             2023-01-01 20:11:00
       2354             2023-01-01 20:11:00
       1410             2023-01-01 20:11:00
       1111             2023-01-01 20:11:00
       4412             2023-01-01 20:11:00
6      5172             2023-01-01 21:17:00
       2354             2023-01-01 21:17:00
       1410             2023-01-01 21:17:00
       1111             2023-01-01 21:17:00
       4412             2023-01-01 21:17:00
7      5172             2023-01-02 05:34:00
       2354             2023-01-02 05:34:00
       1410             2023-01-02 05:34:00
       1111             2023-01-02 05:34:00
       4412             2023-01-02 05:34:00
5 keys hidden

""")
        # [Val_id cardnum.cardnum || tstart.tstart]
        #  1      5172            || 2023-01-01 09:50:00
        #  1      2354            || 2023-01-01 09:50:00
        #  1      1410            || 2023-01-01 09:50:00
        #  1      1111            || 2023-01-01 09:50:00
        #  2      2354            || 2023-01-01 11:10:00
        # ....
        #  7      5172            || 2023-01-02 05:34:00
        #  7      2354            || 2023-01-02 05:34:00
        #  7      1410            || 2023-01-02 05:34:00
        #  7      1111            || 2023-01-02 05:34:00

    def test_ex2_goal4_step3_hide(self):
        # Ah, just throw cardnum away
        # (semantics - if you hide a key that's weak for all values, delete?)
        # (or explicit, guarded delete?)
        s = self.initialise()
        t31 = s.get(["Val_id", "cardnum.cardnum"])
        t32 = t31.infer(["Val_id"], "tstart.tstart")
        t33 = t32.hide("cardnum.cardnum")
        self.assertExpectedInline(str(t33), """\
[Val_id || tstart.tstart]
              tstart.tstart
Val_id                     
1       2023-01-01 09:50:00
2       2023-01-01 11:10:00
3       2023-01-01 15:32:00
4       2023-01-01 15:34:00
5       2023-01-01 20:11:00
6       2023-01-01 21:17:00
7       2023-01-02 05:34:00
1 keys hidden

""")
                #.delete(["cardnum.cardnum"])
        # [Val_id || tstart.tstart]
        #  1      || 2023-01-01 09:50:00
        #  2      || 2023-01-01 11:10:00
        #  3      || 2023-01-01 15:32:00
        #  4      || 2023-01-01 15:34:00
        #  5      || 2023-01-01 20:11:00
        #  6      || 2023-01-01 21:17:00
        #  7      || 2023-01-02 05:34:00

