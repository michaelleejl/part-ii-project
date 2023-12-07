import expecttest
import pandas as pd

from schema import Schema, SchemaNode


class TestEx2(expecttest.TestCase):
    def test_ex2(self):
        s = Schema()
        cardnum = pd.read_csv("./csv/bonuses/cardnum.csv").set_index("val_id")
        tstart = pd.read_csv("./csv/bonuses/tstart.csv").set_index("val_id")
        s.insert_dataframe(cardnum, "cardnum")
        s.insert_dataframe(tstart, "tstart")

        c_cardnum = SchemaNode("cardnum", cluster="cardnum")
        c_val_id = SchemaNode("val_id", cluster="cardnum")
        t_val_id = SchemaNode("val_id", cluster="tstart")
        t_tstart = SchemaNode("tstart", cluster="tstart")

        s.blend(c_val_id, t_val_id, under="Val_id")


        # ========================================================================
        # ========================================================================

        # SCHEMA:
        # cardnum <--- val_id ---> t_start

        # GOAL: For each trip (val_id), tell me when the trip started (t_start)
        # Only two obvious ways to do it

        # WAY 1
        t1 = s.get([t_val_id])
        # Get every val_id in the tstart csv
        # [tstart.val_id || ]
        #  1
        #  2
        #  3
        #  4
        #  5
        #  6
        #  7

        t2 = t1.infer(["tstart.val_id"], t_tstart)
        # [tstart.val_id || tstart.tstart]
        #  1             || 2023-01-01 09:50:00
        #  2             || 2023-01-01 11:10:00
        #  3             || 2023-01-01 15:32:00
        #  4             || 2023-01-01 15:34:00
        #  5             || 2023-01-01 20:11:00
        #  6             || 2023-01-01 21:17:00
        #  7             || 2023-01-02 05:34:00

        # WAY 2
        t11 = s.get([SchemaNode("Val_id")])
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

        t12 = t11.infer(["Val_id"], t_tstart)
        # Values populate keys. Since we use the same values, we will end up with the same keys.
        # [Val_id || tstart.tstart]
        #  1      || 2023-01-01 09:50:00
        #  2      || 2023-01-01 11:10:00
        #  3      || 2023-01-01 15:32:00
        #  4      || 2023-01-01 15:34:00
        #  5      || 2023-01-01 20:11:00
        #  6      || 2023-01-01 21:17:00
        #  7      || 2023-01-02 05:34:00

        # WAY 3
        # Hey, I also want the cardnum, not just the val_id
        # First two steps are the same as either Way 1 or Way 2, but then you can do
        t23 = t12.infer(["Val_id"], c_cardnum)

        # [Val_id || tstart.tstart            cardnum.cardnum]
        #  1      || 2023-01-01 09:50:00      5172
        #  2      || 2023-01-01 11:10:00      2354
        #  3      || 2023-01-01 15:32:00      1410
        #  4      || 2023-01-01 15:34:00      1111
        #  5      || 2023-01-01 20:11:00      2354
        #  6      || 2023-01-01 21:17:00      NA
        #  7      || 2023-01-02 05:34:00      NA

        # Note the presence of NA values! Values populate keys:
        # A key exists as long as ONE value is not NA.

        # STRESS TEST

        # What if the user accidentally makes cardnum a key, so gets the cross product?
        t31 = s.get([t_val_id, c_cardnum])
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

        # Oh... But values populate keys, maybe inference will help a bit?
        # Helps a bit, but not much
        # cardnum is a weak key for tstart, but I think it will be too surprising to suddenly drop it.
        # what if the user is keeping it around for something else?
        t32 = t31.infer(["Val_id"], t_tstart)
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

        # Ah, just throw cardnum away
        # (semantics - if you hide a key that's weak for all values, delete?)
        # (or explicit, guarded delete?)
        t33 = t32.hide(["cardnum.cardnum"])
                #.delete(["cardnum.cardnum"])
        # [Val_id || tstart.tstart]
        #  1      || 2023-01-01 09:50:00
        #  2      || 2023-01-01 11:10:00
        #  3      || 2023-01-01 15:32:00
        #  4      || 2023-01-01 15:34:00
        #  5      || 2023-01-01 20:11:00
        #  6      || 2023-01-01 21:17:00
        #  7      || 2023-01-02 05:34:00

