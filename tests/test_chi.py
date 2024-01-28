import expecttest
import numpy as np
import pandas as pd

from schema import Schema, SchemaNode, Cardinality
from tables.function import create_function


def initialise():
    s = Schema()
    flows = pd.read_csv("./csv/flows/flows.csv").set_index(["from_city", "to_city"])
    flows = s.insert_dataframe(flows)
    City = s.create_class("City")
    s.blend(flows["from_city"], flows["to_city"], City)
    from_city = flows["from_city"]
    to_city = flows["to_city"]
    volume = flows["volume"]
    return s, City, from_city, to_city, volume


class TestChi(expecttest.TestCase):


    def test1(self):
        s, City, from_city, to_city, volume = initialise()

        t0 = s.get([from_city, to_city])
        t01 = t0.infer(["from_city", "to_city"], volume)
        self.assertExpectedInline(str(t01), """\
[from_city to_city || volume]
                     volume
from_city to_city          
Cambridge London        3.0
          Edinburgh     2.4
          Oxford        0.6
London    Cambridge     4.2
          Oxford        2.4
Oxford    London        1.2
          Edinburgh     1.8
Edinburgh Cambridge     1.8
          London        2.4
7 keys hidden

""")

    def test2(self):
        s, City, from_city, to_city, volume = initialise()

        t0 = s.get([City, City], ["FromCity", "ToCity"])
        t01 = t0.infer(["FromCity", "ToCity"], volume)
        self.assertExpectedInline(str(t01), """\
[FromCity ToCity || volume]
                     volume
FromCity  ToCity           
Cambridge London        3.0
          Edinburgh     2.4
          Oxford        0.6
London    Cambridge     4.2
          Oxford        2.4
Oxford    London        1.2
          Edinburgh     1.8
Edinburgh Cambridge     1.8
          London        2.4
7 keys hidden

""")

# TODO: Register function
    def test3(self):
        s, City, from_city, to_city, volume = initialise()
        t1 = s.get([City, City], ["FromCity", "ToCity"])
        t2 = t1.infer(["FromCity", "ToCity"], volume)
        t3 = t2.extend("volume", 0, "volume_fillna").sort(["FromCity", "ToCity"])
        self.assertExpectedInline(str(t3), """\
[FromCity ToCity || volume volume_fillna]
                     volume  volume_fillna
FromCity  ToCity                          
Cambridge Cambridge     NaN            0.0
          Edinburgh     2.4            2.4
          London        3.0            3.0
          Oxford        0.6            0.6
Edinburgh Cambridge     1.8            1.8
          Edinburgh     NaN            0.0
          London        2.4            2.4
          Oxford        NaN            0.0
London    Cambridge     4.2            4.2
          Edinburgh     NaN            0.0
          London        NaN            0.0
          Oxford        2.4            2.4
Oxford    Cambridge     NaN            0.0
          Edinburgh     1.8            1.8
          London        1.2            1.2
          Oxford        NaN            0.0

""")

    def test4(self):
        s, City, from_city, to_city, volume = initialise()
        t1 = s.get([City, City], ["FromCity", "ToCity"])
        t2 = t1.infer(["FromCity", "ToCity"], volume)
        t3 = t2.extend("volume", 0, "volume_fillna")
        t4 = t3.swap("FromCity", "ToCity").sort(["FromCity", "ToCity"])
        self.assertExpectedInline(str(t4), """\
[ToCity FromCity || volume volume_fillna]
                     volume  volume_fillna
ToCity    FromCity                        
Cambridge Cambridge     NaN            0.0
Edinburgh Cambridge     2.4            2.4
London    Cambridge     3.0            3.0
Oxford    Cambridge     0.6            0.6
Cambridge Edinburgh     1.8            1.8
Edinburgh Edinburgh     NaN            0.0
London    Edinburgh     2.4            2.4
Oxford    Edinburgh     NaN            0.0
Cambridge London        4.2            4.2
Edinburgh London        NaN            0.0
London    London        NaN            0.0
Oxford    London        2.4            2.4
Cambridge Oxford        NaN            0.0
Edinburgh Oxford        1.8            1.8
London    Oxford        1.2            1.2
Oxford    Oxford        NaN            0.0

""")

    def test5(self):
        s, City, from_city, to_city, volume = initialise()
        t1 = s.get([City, City], ["FromCity", "ToCity"])
        t2 = t1.infer(["FromCity", "ToCity"], volume)
        t3 = t2.extend("volume", 0, "volume_fillna")
        t4 = t3.swap("FromCity", "ToCity")
        t5 = t4.shift_left()
        self.assertExpectedInline(str(t5), """\
[ToCity || FromCity volume volume_fillna]
                                         FromCity  ...         volume_fillna
ToCity                                             ...                      
London     [Cambridge, London, Edinburgh, Oxford]  ...  [3.0, 0.0, 2.4, 1.2]
Edinburgh  [Cambridge, London, Edinburgh, Oxford]  ...  [2.4, 0.0, 0.0, 1.8]
Oxford     [Cambridge, London, Edinburgh, Oxford]  ...  [0.6, 2.4, 0.0, 0.0]
Cambridge  [Cambridge, London, Edinburgh, Oxford]  ...  [0.0, 4.2, 1.8, 0.0]

[4 rows x 3 columns]

""")

    def test6(self):
        s, City, from_city, to_city, volume = initialise()
        t1 = s.get([City, City], ["FromCity", "ToCity"])
        t2 = t1.infer(["FromCity", "ToCity"], volume)
        t3 = t2.extend("volume", 0, "volume_fillna")
        t4 = t3.swap("FromCity", "ToCity")
        t5 = t4.shift_left()
        t6 = t5.deduce(t5["volume_fillna"].sum(), "total_inflow")
        self.assertExpectedInline(str(t6), """\
[ToCity || FromCity volume volume_fillna total_inflow]
                                         FromCity  ... total_inflow
ToCity                                             ...             
London     [Cambridge, Edinburgh, Oxford, London]  ...          6.6
Edinburgh  [Cambridge, Edinburgh, Oxford, London]  ...          4.2
Oxford     [Cambridge, Edinburgh, Oxford, London]  ...          3.0
Cambridge  [Cambridge, Edinburgh, Oxford, London]  ...          6.0

[4 rows x 4 columns]

""")

    def test7(self):
        s, City, from_city, to_city, volume = initialise()
        t1 = s.get([City, City], ["FromCity", "ToCity"])
        t2 = t1.infer(["FromCity", "ToCity"], volume)
        t3 = t2.extend("volume", 0, "volume_fillna")
        t4 = t3.swap("FromCity", "ToCity")
        t5 = t4.shift_left()
        t6 = t5.deduce(t5["volume_fillna"].sum(), "total_inflow")
        t7 = t6.deduce(t6["volume_fillna"]/t6["total_inflow"], "relative_inflow")
        self.assertExpectedInline(str(t7), """\
[ToCity || FromCity volume volume_fillna total_inflow relative_inflow]
                                         FromCity  ...                                    relative_inflow
ToCity                                             ...                                                   
London     [Cambridge, Edinburgh, Oxford, London]  ...  [0.45454545454545453, 0.3636363636363636, 0.18...
Edinburgh  [Cambridge, Edinburgh, Oxford, London]  ...  [0.5714285714285714, 0.0, 0.42857142857142855,...
Oxford     [Cambridge, Edinburgh, Oxford, London]  ...  [0.19999999999999998, 0.0, 0.0, 0.799999999999...
Cambridge  [Cambridge, Edinburgh, Oxford, London]  ...                [0.0, 0.3, 0.0, 0.7000000000000001]

[4 rows x 5 columns]

""")
    #     t5 = t4b.assign("total_inflows", t4b["total_inflow"].aggregate(sum))
    #     print(t5)
    #
    #
    #     ## compute total outflow
    #     t6 = t3.set_key(["FromCity"])
    #     print(t6)
    #     t7 = t6.assign("total_outflow", t6["volume"].aggregate(sum)).sort(["FromCity", "ToCity"])
    #     print(t7)
    #
    #
    #
    #     ## combine them back into the same table
    #     t8 = t3.infer(["ToCity"], "total_inflow")
    #     print(t8)
    #     t9 = t8.infer(["FromCity"], "total_outflow")
    #     print(t9)
    #
    #
    #     # t10 = (t9.assign("relative_outflow", t9["volume"] / t9["total_outflow"])
    #     #          .assign("expected_outflow", t9["total_inflow"] / t9["total_inflows"])
    #     #          .set_key(["FromCity"]).hide("volume").hide("total_outflow").hide("ToCity").hide("total_inflow").hide("total_inflows"))
    #     # print(t10)
    #
    # # print(t5)


