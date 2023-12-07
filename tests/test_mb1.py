import expecttest
import pandas as pd

from schema import Schema, SchemaNode


def latlng(lat, lng):
    pass


class TestMb1(expecttest.TestCase):

    def test_mb_1(self):
        s = Schema()
        commutes = pd.read_csv("./csv/bilhetagem/commutes.csv").set_index("val_id")
        s.insert_dataframe(commutes, "commutes")
        val_id = SchemaNode("val_id", cluster="commutes")
        lat = SchemaNode("lat", cluster="commutes")
        lng = SchemaNode("lng", cluster="commutes")
        bus_line = SchemaNode("bus_line", cluster="commutes")
        date = SchemaNode("date", cluster="commutes")
        card_id = SchemaNode("card_id", cluster="commutes")

        t1 = s.get([val_id]).infer([val_id], [bus_line, lat, lng, date, card_id])
        # [val_id || bus_line lat   lng   date    card_id]
        #  1      || 10       38.7  -9.2  09:00   001
        #  2      || 12       41.1  -8.6  09:10   001
        #  3      || 13       41.4  -8.3  09:11   002
        #  4      || 12       41.5  -8.4  15:00   002


        # We need to wrap the SchemaNode into
        # TODO: what do we do about
        # TODO:
        t1["loc"] =  latlng(t1["lat"], t1["lng"])
        # [val_id || bus_line lat   lng   loc  date    card_id]
        #  1      || 10       38.7  -9.2  l1   09:00   001
        #  2      || 12       41.1  -8.6  l2   09:10   001
        #  3      || 13       41.4  -8.3  l3   09:11   002
        #  4      || 12       41.5  -8.4  l4   15:00   002

        t2 = t1.hide_val(["lat", "lng"])
        # [val_id || bus_line  loc  date    card_id]
        #  1      || 10        l1   09:00   001
        #  2      || 12        l2   09:10   001
        #  3      || 13        l3   09:11   002
        #  4      || 12        l4   15:00   002

        # for all val columns, the hidden key is val_id.
        t3 = t2.set_key(["card_id"])
        # [card_id || bus_line  loc       date            val_id]
        #  001     || [10,12]   [l1,l2]   [09:00,09:10]   [1,2]
        #  002     || [13,12]   [l3,l4]   [09:11,15:00]   [3,4]

        def apply_commute_rules(x):
            return x

        t3["commute_id"] = t3["val_id"].map(apply_commute_rules)
        # [card_id || bus_line  loc       date            val_id   commute_id]
        #  001     || [10,12]   [l1,l2]   [09:00,09:10]   [1,2]    [1,1]
        #  002     || [13,12]   [l3,l4]   [09:11,15:00]   [3,4]    [2,3]

        t4 = t3.set_key(["commute_id"])
        # [commute_id || bus_line  loc       date            val_id   card_id]
        #  1          || [10,12]   [l1,l2]   [09:00,09:10]   [1,2]    [001, 001]
        #  2          || [13]      [l3]      [09:11]         [3]      [002]
        #  3          || [12]      [l4]      [15:00]         [4]      [002]

        t4["last_val"] = last_val(t3["val_id"])
        # [commute_id || bus_line  loc       date            val_id   card_id       last_val]
        #  1          || [10,12]   [l1,l2]   [09:00,09:10]   [1,2]    [001, 001]    2
        #  2          || [13]      [l3]      [09:11]         [3]      [002]         3
        #  3          || [12]      [l4]      [15:00]         [4]      [002]         4

        t5 = t4.set_key("last_val")["last_val", "commute_id"]
        # [last_val || commute_id]
        #  2        || 1
        #  3        || 2
        #  4        || 3

        t6 = t5.combine(t1, t5)
        # [val_id last_val || bus_line loc  date    card_id  commute_id]
        #  1      2        || 10       l1   09:00   001      1
        #  1      3        || 10       l1   09:00   001      2
        #  1      4        || 10       l1   09:00   001      3
        #  2      2        || 12       l2   09:10   001      1
        #  2      3        || 12       l2   09:10   001      2
        #  2      4        || 12       l2   09:10   001      3
        #  3      2        || 13       l3   09:11   002      1
        #  3      3        || 13       l3   09:11   002      2
        #  3      4        || 13       l3   09:11   002      3
        #  4      2        || 12       l4   15:00   002      1
        #  4      3        || 12       l4   15:00   002      2
        #  4      4        || 12       l4   15:00   002      3

        t7 = t6.equate("last_val", "val_id")
        # [last_val || bus_line   loc  date    card_id  commute_id]
        #  2        || 12         l2   09:10   001      1
        #  3        || 13         l3   09:11   002      2
        #  4        || 12         l4   15:00   002      3

        t8 = t7.set_key(["bus_line"])
        # [bus_line || last_val  loc        date             card_id        commute_id]
        #  12       || [2,4]     [l2, l4]   [09:10,15:00]    [001,002]      [1,3]
        #  13       || [3]       [l3]       [09:11]          [002]          [2]

        t8["last_mile_trips"] = count(t8["commute_id"])
        # [bus_line || last_val  loc        date             card_id        commute_id   last_mile_trips]
        #  12       || [2,4]     [l2, l4]   [09:10,15:00]    [001,002]      [1,3]        2
        #  13       || [3]       [l3]       [09:11]          [002]          [2]          1
