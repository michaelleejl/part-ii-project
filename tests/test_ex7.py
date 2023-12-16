import expecttest
import pandas as pd

from schema import Schema, SchemaNode


class TestEx7(expecttest.TestCase):

    def initialise(self) -> Schema:
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
        return s

        # SCHEMA:
        # cardnum <--- val_id ---> t_start
        # val_id, cardnum ---> bonus

    # First, get [val_id || cardnum]
    def test_ex7_goal1_step1_get(self):
        # Get every Val_id, Cardnum pair
        s = self.initialise()
        t1 = s.get(["cardnum.val_id"]).infer(["cardnum.val_id"], "cardnum.cardnum")
        self.assertExpectedInline(str(t1), """\
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
        # [cardnum.val_id || cardnum.cardnum ]
        #  1              || 5172
        #  2              || 2354
        #  3              || 1410
        #  4              || 1111
        #  5              || 2354
        #  8              || 4412

    def test_ex7_goal1_step2_infer(self):
        s = self.initialise()
        t1 = s.get(["cardnum.val_id"]).infer(["cardnum.val_id"], "cardnum.cardnum")
        t2 = t1.infer(["cardnum.val_id"], "bonus.bonus")
        self.maxDiff = None
        self.assertExpectedInline(str(t2), """\
[cardnum.val_id || cardnum.cardnum bonus.bonus]
                cardnum.cardnum  bonus.bonus
cardnum.val_id                              
1                          5172  [4.0, 12.0]
2                          2354   [5.0, 7.0]
3                          1410        [1.0]
5                          2354        [2.0]
4                          1111           []
8                          4412           []

""")
        # [cardnum.val_id || cardnum.cardnum bonus.bonus]
        #  1              || 5172            [4, 12]
        #  2              || 2354            [5, 7]
        #  3              || 1410            [1]
        #  4              || 1111            []
        #  5              || 2354            [2]
        #  8              || 4412            []

    def test_ex7_goal1_step3_setKey(self):
        s = self.initialise()
        t1 = s.get(["cardnum.val_id"]).infer(["cardnum.val_id"], "cardnum.cardnum")
        t2 = t1.infer(["cardnum.val_id"], "bonus.bonus")
        t3 = t2.set_key(["cardnum.val_id", "cardnum.cardnum"])
        self.assertExpectedInline(str(t3), """\
[cardnum.val_id cardnum.cardnum || bonus.bonus]
                                bonus.bonus
cardnum.val_id cardnum.cardnum             
1              5172             [4.0, 12.0]
2              2354              [5.0, 7.0]
3              1410                   [1.0]
5              2354                   [2.0]
2 keys hidden

""")

        # [cardnum.val_id cardnum.cardnum || bonus.bonus]
        #  1              5172            || [4, 12]
        #  2              2354            || [5, 7]
        #  3              1410            || [1]
        #  5              2354            || [2]

        # "Now, only show the k, x levels supported by y, and ignore the
        #  old k levels supported (only) by x"

    def test_ex7_goal1_step4_show(self):
        s = self.initialise()
        t1 = s.get(["cardnum.val_id"]).infer(["cardnum.val_id"], "cardnum.cardnum")
        t2 = t1.infer(["cardnum.val_id"], "bonus.bonus")
        t3 = t2.set_key(["cardnum.val_id", "cardnum.cardnum"])
        t4 = t3.show("bonus.cardnum")
        self.assertExpectedInline(str(t4), """\
[cardnum.val_id cardnum.cardnum bonus.cardnum || bonus.bonus]
                                              bonus.bonus
cardnum.val_id cardnum.cardnum bonus.cardnum             
1              5172            5172.0                 4.0
                               1410.0                12.0
2              2354            1111.0                 5.0
                               6440.0                 7.0
3              1410            1111.0                 1.0
5              2354            1410.0                 2.0
2 keys hidden

""")

        # [cardnum.val_id cardnum.cardnum bonus.cardnum  || bonus.bonus]
        #  1              5172            5172           || 4
        #  1              5172            1410           || 12
        #  2              2354            1111           || 5
        #  2              2354            6440           || 7
        #  3              1410            1111           || 1
        #  5              2354            1410           || 2

    def test_ex7_goal1_step5_equate(self):
        # # Inner product
        s = self.initialise()
        t1 = s.get(["cardnum.val_id"]).infer(["cardnum.val_id"], "cardnum.cardnum")
        t2 = t1.infer(["cardnum.val_id"], "bonus.bonus")
        t3 = t2.set_key(["cardnum.val_id", "cardnum.cardnum"])
        t4 = t3.show("bonus.cardnum")
        t5 = t4.equate("cardnum.cardnum", "bonus.cardnum")
        self.assertExpectedInline(str(t5), """\
[cardnum.val_id cardnum.cardnum || bonus.bonus]
                                bonus.bonus
cardnum.val_id cardnum.cardnum             
1              5172                     4.0

""")
        # # [cardnum.val_id cardnum.cardnum  || bonus.bonus]
        # #  1              5172             || 4

        # Not the same as filter! Because it changes the strength of keys.
        # Now anything that depended on bonus.cardnum depends on cardnum.cardnum