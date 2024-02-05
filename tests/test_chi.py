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
        self.maxDiff = None
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
        self.maxDiff = None
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

    def test8(self):
        s, City, from_city, to_city, volume = initialise()
        t1 = s.get([City, City], ["FromCity", "ToCity"])
        t2 = t1.infer(["FromCity", "ToCity"], volume)
        t3 = t2.extend("volume", 0, "volume_fillna")
        t4 = t3.swap("FromCity", "ToCity")
        t5 = t4.shift_left()
        t6 = t5.deduce(t5["volume_fillna"].sum(), "total_inflow")
        t7 = t6.deduce(t6["volume_fillna"] / t6["total_inflow"], "relative_inflow")
        t8 = t2.infer(["ToCity"], t7["relative_inflow"])
        #TODO
        self.assertExpectedInline(str(t8), """\
[FromCity ToCity || volume relative_inflow]
                     volume                                    relative_inflow
FromCity  ToCity                                                              
Cambridge London        3.0                [1.0, 1.5714285714285714, 2.2, 1.1]
          Edinburgh     2.4  [0.6363636363636364, 1.0, 1.4000000000000001, ...
          Oxford        0.6  [0.45454545454545453, 0.7142857142857143, 1.0,...
London    Cambridge     4.2  [0.9090909090909091, 1.4285714285714286, 2.0, ...
          Oxford        2.4  [0.45454545454545453, 0.7142857142857143, 1.0,...
Oxford    London        1.2                [1.0, 1.5714285714285714, 2.2, 1.1]
          Edinburgh     1.8  [0.6363636363636364, 1.0, 1.4000000000000001, ...
Edinburgh Cambridge     1.8  [0.9090909090909091, 1.4285714285714286, 2.0, ...
          London        2.4                [1.0, 1.5714285714285714, 2.2, 1.1]
Cambridge Cambridge     NaN  [0.9090909090909091, 1.4285714285714286, 2.0, ...
London    London        NaN                [1.0, 1.5714285714285714, 2.2, 1.1]
          Edinburgh     NaN  [0.6363636363636364, 1.0, 1.4000000000000001, ...
Oxford    Cambridge     NaN  [0.9090909090909091, 1.4285714285714286, 2.0, ...
          Oxford        NaN  [0.45454545454545453, 0.7142857142857143, 1.0,...
Edinburgh Oxford        NaN  [0.45454545454545453, 0.7142857142857143, 1.0,...
          Edinburgh     NaN  [0.6363636363636364, 1.0, 1.4000000000000001, ...

""")

    def test9(self):
        s, City, from_city, to_city, volume = initialise()
        t1 = s.get([City, City], ["FromCity", "ToCity"])
        t2 = t1.infer(["FromCity", "ToCity"], volume)
        t3 = t2.extend("volume", 0, "volume_fillna")
        t9 = t3.shift_left()
        self.assertExpectedInline(str(t9), """\
[FromCity || ToCity volume volume_fillna]
                                           ToCity  ...         volume_fillna
FromCity                                           ...                      
Cambridge  [London, Edinburgh, Oxford, Cambridge]  ...  [3.0, 2.4, 0.6, 0.0]
Edinburgh  [London, Edinburgh, Oxford, Cambridge]  ...  [2.4, 0.0, 0.0, 1.8]
Oxford     [London, Edinburgh, Oxford, Cambridge]  ...  [1.2, 1.8, 0.0, 0.0]
London     [London, Edinburgh, Oxford, Cambridge]  ...  [0.0, 0.0, 2.4, 4.2]

[4 rows x 3 columns]

""")

    def test10(self):
        s, City, from_city, to_city, volume = initialise()
        t1 = s.get([City, City], ["FromCity", "ToCity"])
        t2 = t1.infer(["FromCity", "ToCity"], volume)
        t3 = t2.extend("volume", 0, "volume_fillna")
        t9 = t3.shift_left()
        t10 = t9.deduce(t9["volume_fillna"].sum(), "total_outflow")
        self.assertExpectedInline(str(t10), """\
[FromCity || ToCity volume volume_fillna total_outflow]
                                           ToCity  ... total_outflow
FromCity                                           ...              
Cambridge  [London, Edinburgh, Oxford, Cambridge]  ...           6.0
London     [London, Edinburgh, Oxford, Cambridge]  ...           6.6
Edinburgh  [London, Edinburgh, Oxford, Cambridge]  ...           4.2
Oxford     [London, Edinburgh, Oxford, Cambridge]  ...           3.0

[4 rows x 4 columns]

""")

    def test11(self):
        s, City, from_city, to_city, volume = initialise()
        t1 = s.get([City, City], ["FromCity", "ToCity"])
        t2 = t1.infer(["FromCity", "ToCity"], volume)
        t3 = t2.extend("volume", 0, "volume_fillna")
        t9 = t3.shift_left()
        t10 = t9.deduce(t9["volume_fillna"].sum(), "total_outflow")
        t11 = t10.deduce(t10["volume_fillna"] / t10["total_outflow"], "relative_outflow")
        self.assertExpectedInline(str(t11), """\
[FromCity || ToCity volume volume_fillna total_outflow relative_outflow]
                                           ToCity  ...                                   relative_outflow
FromCity                                           ...                                                   
Cambridge  [London, Edinburgh, Oxford, Cambridge]  ...  [0.5, 0.39999999999999997, 0.09999999999999999...
London     [London, Edinburgh, Oxford, Cambridge]  ...  [0.0, 0.0, 0.36363636363636365, 0.636363636363...
Edinburgh  [London, Edinburgh, Oxford, Cambridge]  ...  [0.5714285714285714, 0.0, 0.0, 0.4285714285714...
Oxford     [London, Edinburgh, Oxford, Cambridge]  ...               [0.39999999999999997, 0.6, 0.0, 0.0]

[4 rows x 5 columns]

""")

    def test12(self):
        s, City, from_city, to_city, volume = initialise()
        t1 = s.get([City, City], ["FromCity", "ToCity"])
        t2 = t1.infer(["FromCity", "ToCity"], volume)
        t3 = t2.extend("volume", 0, "volume_fillna")
        t4 = t3.swap("FromCity", "ToCity")
        t5 = t4.shift_left()
        t6 = t5.deduce(t5["volume_fillna"].sum(), "total_inflow")
        t7 = t6.deduce(t6["volume_fillna"] / t6["total_inflow"], "relative_inflow")

        t9 = t3.shift_left()
        t10 = t9.deduce(t9["volume_fillna"].sum(), "total_outflow")
        t11 = t10.deduce(t10["volume_fillna"] / t10["total_outflow"], "relative_outflow")

        t12 = t2.infer(["ToCity", "FromCity"], t7["relative_inflow"], with_name="expected_outflow")
        print("T12 DERIVATION")
        print(t12.derivation)
        self.assertExpectedInline(str(t12), """\
[FromCity ToCity || volume expected_outflow]
                     volume                                   expected_outflow
FromCity  ToCity                                                              
Cambridge London        3.0                [1.0, 1.5714285714285714, 2.2, 1.1]
London    London        NaN                [1.0, 1.5714285714285714, 2.2, 1.1]
Oxford    London        1.2                [1.0, 1.5714285714285714, 2.2, 1.1]
Edinburgh London        2.4                [1.0, 1.5714285714285714, 2.2, 1.1]
Cambridge Edinburgh     2.4  [0.6363636363636364, 1.0, 1.4000000000000001, ...
London    Edinburgh     NaN  [0.6363636363636364, 1.0, 1.4000000000000001, ...
Oxford    Edinburgh     1.8  [0.6363636363636364, 1.0, 1.4000000000000001, ...
Edinburgh Edinburgh     NaN  [0.6363636363636364, 1.0, 1.4000000000000001, ...
Cambridge Oxford        0.6  [0.45454545454545453, 0.7142857142857143, 1.0,...
London    Oxford        2.4  [0.45454545454545453, 0.7142857142857143, 1.0,...
Oxford    Oxford        NaN  [0.45454545454545453, 0.7142857142857143, 1.0,...
Edinburgh Oxford        NaN  [0.45454545454545453, 0.7142857142857143, 1.0,...
Cambridge Cambridge     NaN  [0.9090909090909091, 1.4285714285714286, 2.0, ...
London    Cambridge     4.2  [0.9090909090909091, 1.4285714285714286, 2.0, ...
Oxford    Cambridge     NaN  [0.9090909090909091, 1.4285714285714286, 2.0, ...
Edinburgh Cambridge     1.8  [0.9090909090909091, 1.4285714285714286, 2.0, ...

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


