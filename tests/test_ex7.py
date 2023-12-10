import expecttest
import pandas as pd

from schema import Schema, SchemaNode


class TestEx7(expecttest.TestCase):
    def test_ex7(self):
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

        s.blend(c_val_id, t_val_id, under ="Val_id")
        s.blend(c_val_id, b_val_id)
        s.blend(c_cardnum, b_cardnum, under ="Cardnum")

        # ========================================================================
        # ========================================================================

        # SCHEMA:
        # cardnum <--- val_id ---> t_start
        # val_id, cardnum ---> bonus

        # First, get [val_id || cardnum]

        # Get every Val_id, Cardnum pair
        t1 = s.get(["cardnum.val_id"]).infer(["cardnum.val_id"], "cardnum.cardnum")
        print(t1)
        # [cardnum.val_id || cardnum.cardnum ]
        #  1              || 5172
        #  2              || 2354
        #  3              || 1410
        #  4              || 1111
        #  5              || 2354
        #  8              || 4412


        t2 = t1.infer(["cardnum.val_id"], "bonus.bonus")
        print(t2)
        # [cardnum.val_id || cardnum.cardnum bonus.bonus]
        #  1              || 5172            [4, 12]
        #  2              || 2354            [5, 7]
        #  3              || 1410            [1]
        #  4              || 1111            []
        #  5              || 2354            [2]
        #  8              || 4412            []

        t3 = t2.set_key(["cardnum.val_id", "cardnum.cardnum"])
        print(t3)

        # [cardnum.val_id cardnum.cardnum || bonus.bonus]
        #  1              5172            || [4, 12]
        #  2              2354            || [5, 7]
        #  3              1410            || [1]
        #  5              2354            || [2]

        # "Now, only show the k, x levels supported by y, and ignore the
        #  old k levels supported (only) by x"

        t4 = t3.show("bonus.cardnum")
        print(t4)

        # [cardnum.val_id cardnum.cardnum bonus.cardnum  || bonus.bonus]
        #  1              5172            5172           || 4
        #  1              5172            1410           || 12
        #  2              2354            1111           || 5
        #  2              2354            6440           || 7
        #  3              1410            1111           || 1
        #  5              2354            1410           || 2

        # # Inner product time
        # t5 = t4.equate("cardnum.cardnum", "bonus.cardnum")
        # # [cardnum.val_id cardnum.cardnum  || bonus.bonus]
        # #  1              5172             || 4

        # Not the same as filter! Because it changes the strength of keys.
        # Now anything that depended on bonus.cardnum depends on cardnum.cardnum