import expecttest
import pandas as pd

from schema import Schema, SchemaNode


class TestEx7(expecttest.TestCase):

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

        Val_id = s.create_class("Val_id")
        Cardnum = s.create_class("Cardnum")

        s.blend(c_val_id, t_val_id, Val_id)
        s.blend(c_val_id, b_val_id)
        s.blend(c_cardnum, b_cardnum, Cardnum)
        return s, bonus, cardnum, tstart, Cardnum, Val_id

        # SCHEMA:
        # cardnum <--- val_id ---> t_start
        # val_id, cardnum ---> bonus

    # First, get [val_id || cardnum]
    def test_ex7_goal1_step1_get(self):
        # Get every Val_id, Cardnum pair
        s, bonus, cardnum, tstart, Cardnum, Val_id = self.initialise()
        t1 = s.get([cardnum["val_id"]]).infer(["val_id"], cardnum["cardnum"])
        self.assertExpectedInline(
            str(t1),
            """\
[val_id || cardnum]
        cardnum
val_id         
1          5172
2          2354
3          1410
4          1111
5          2354
8          4412

""",
        )
        # [cardnum.val_id || cardnum.cardnum ]
        #  1              || 5172
        #  2              || 2354
        #  3              || 1410
        #  4              || 1111
        #  5              || 2354
        #  8              || 4412

    def test_ex7_goal1_step2_infer(self):
        s, bonus, cardnum, tstart, Cardnum, Val_id = self.initialise()
        t1 = s.get([cardnum["val_id"]]).infer(["val_id"], cardnum["cardnum"])
        t2 = t1.infer(["val_id"], bonus["bonus"])
        self.maxDiff = None
        self.assertExpectedInline(
            str(t2),
            """\
[val_id || cardnum bonus]
        cardnum        bonus
val_id                      
1          5172  [4.0, 12.0]
2          2354   [5.0, 7.0]
3          1410        [1.0]
5          2354        [2.0]
4          1111           []
8          4412           []

""",
        )
        # [cardnum.val_id || cardnum.cardnum bonus.bonus]
        #  1              || 5172            [4, 12]
        #  2              || 2354            [5, 7]
        #  3              || 1410            [1]
        #  4              || 1111            []
        #  5              || 2354            [2]
        #  8              || 4412            []

    def test_ex7_goal1_step3_setKey(self):
        s, bonus, cardnum, tstart, Cardnum, Val_id = self.initialise()
        t1 = s.get([cardnum["val_id"]]).infer(["val_id"], cardnum["cardnum"])
        t2 = t1.infer(["val_id"], bonus["bonus"])
        t3 = t2.shift_right()
        self.assertExpectedInline(
            str(t3),
            """\
[val_id cardnum || bonus]
                      bonus
val_id cardnum             
1      5172     [4.0, 12.0]
2      2354      [5.0, 7.0]
3      1410           [1.0]
5      2354           [2.0]
26 keys hidden

""",
        )

        # [cardnum.val_id cardnum.cardnum || bonus.bonus]
        #  1              5172            || [4, 12]
        #  2              2354            || [5, 7]
        #  3              1410            || [1]
        #  5              2354            || [2]

        # "Now, only show the k, x levels supported by y, and ignore the
        #  old k levels supported (only) by x"

    def test_ex7_goal1_step4_show(self):
        s, bonus, cardnum, tstart, Cardnum, Val_id = self.initialise()
        t1 = s.get([cardnum["val_id"]]).infer(["val_id"], cardnum["cardnum"])
        t2 = t1.infer(["val_id"], bonus["bonus"])
        t3 = t2.shift_right()
        t4 = t3.show("cardnum_1")
        self.maxDiff = None
        self.assertExpectedInline(
            str(t4),
            """\
[val_id cardnum cardnum_1 || bonus]
                          bonus
val_id cardnum cardnum_1       
1      5172    5172         4.0
2      2354    1111         5.0
3      1410    1111         1.0
1      5172    1410        12.0
2      2354    6440         7.0
5      2354    1410         2.0
114 keys hidden

""",
        )

        # [cardnum.val_id cardnum.cardnum bonus.cardnum  || bonus.bonus]
        #  1              5172            5172           || 4
        #  1              5172            1410           || 12
        #  2              2354            1111           || 5
        #  2              2354            6440           || 7
        #  3              1410            1111           || 1
        #  5              2354            1410           || 2

    def test_ex7_goal1_step5_equate(self):
        # # Inner product
        s, bonus, cardnum, tstart, Cardnum, Val_id = self.initialise()
        t1 = s.get([cardnum["val_id"]]).infer(["val_id"], cardnum["cardnum"])
        t2 = t1.infer(["val_id"], bonus["bonus"])
        t3 = t2.shift_right()
        t4 = t3.show("cardnum_1")
        t5 = t4.mask(
            "cardnum",
            (t4["cardnum"] == t4["cardnum_1"]) & t4["bonus"].isnotnull(),
            "is_cardnum_equal",
        )
        t6 = t5.filter("is_cardnum_equal")
        self.maxDiff = None
        self.assertExpectedInline(
            str(t6),
            """\
[val_id cardnum cardnum_1 || bonus is_cardnum_equal]
                          bonus  is_cardnum_equal
val_id cardnum cardnum_1                         
1      5172    5172         4.0            5172.0
119 keys hidden

""",
        )
        # # [cardnum.val_id cardnum.cardnum  || bonus.bonus]
        # #  1              5172             || 4

        # Not the same as filter! Because it changes the strength of keys.
        # Now anything that depended on bonus.cardnum depends on cardnum.cardnum
