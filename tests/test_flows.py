import expecttest
import pandas as pd

from schema.schema import Schema


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


class TestFlows(expecttest.TestCase):

    def test(self):
        s, City, from_city, to_city, volume = initialise()

        f = s.get(from_city=from_city, to_city=to_city)
        flows = f.infer(["from_city", "to_city"], volume)
        outf = flows.hide("to_city")
        outf = outf.mutate(outflow=outf["volume"].sum())
        inf = flows.hide("from_city")
        inf = inf.mutate(inflow=inf["volume"].sum())
        join = (
            s.get(city=City)
            .infer(["city"], outf["outflow"])
            .infer(["city"], inf["inflow"])
        )
        netflow = join.mutate(net_outflow=join["outflow"] - join["inflow"])
        self.assertExpectedInline(
            str(netflow),
            """\
[city || outflow inflow net_outflow]
           outflow  inflow  net_outflow
city                                   
London         6.6     6.6          0.0
Edinburgh      4.2     4.2          0.0
Oxford         3.0     3.0          0.0
Cambridge      6.0     6.0          0.0

""",
        )

    def test1(self):
        s, City, from_city, to_city, volume = initialise()

        t0 = s.get(from_city=from_city, to_city=to_city)
        t01 = t0.infer(["from_city", "to_city"], volume)
        self.assertExpectedInline(
            str(t01),
            """\
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

""",
        )

    def test2(self):
        s, City, from_city, to_city, volume = initialise()

        t0 = s.get(FromCity=City, ToCity=City)
        t01 = t0.infer(["FromCity", "ToCity"], volume)
        self.assertExpectedInline(
            str(t01),
            """\
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

""",
        )

    # TODO: Register function
    def test3(self):
        s, City, from_city, to_city, volume = initialise()
        t1 = s.get(FromCity=City, ToCity=City)
        t2 = t1.infer(["FromCity", "ToCity"], volume)
        t3 = t2.mutate(volume_fillna=t2["volume"].extend(0)).sort(
            ["FromCity", "ToCity"]
        )
        self.assertExpectedInline(
            str(t3),
            """\
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

""",
        )

    def test4(self):
        s, City, from_city, to_city, volume = initialise()
        t1 = s.get(FromCity=City, ToCity=City)
        t2 = t1.infer(["FromCity", "ToCity"], volume)
        t3 = t2.mutate(volume_fillna=t2["volume"].extend(0))
        t4 = t3.swap("FromCity", "ToCity").sort(["FromCity", "ToCity"])
        self.assertExpectedInline(
            str(t4),
            """\
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

""",
        )

    def test5(self):
        s, City, from_city, to_city, volume = initialise()
        t1 = s.get(FromCity=City, ToCity=City)
        t2 = t1.infer(["FromCity", "ToCity"], volume)
        t3 = t2.mutate(volume_fillna=t2["volume"].extend(0))
        t4 = t3.swap("FromCity", "ToCity")
        t5 = t4.shift_left()
        self.assertExpectedInline(
            str(t5),
            """\
[ToCity || FromCity volume volume_fillna]
                                         FromCity  ...         volume_fillna
ToCity                                             ...                      
London     [Cambridge, London, Oxford, Edinburgh]  ...  [3.0, 0.0, 1.2, 2.4]
Edinburgh  [Cambridge, London, Oxford, Edinburgh]  ...  [2.4, 0.0, 1.8, 0.0]
Oxford     [Cambridge, London, Oxford, Edinburgh]  ...  [0.6, 2.4, 0.0, 0.0]
Cambridge  [Cambridge, London, Oxford, Edinburgh]  ...  [0.0, 4.2, 0.0, 1.8]

[4 rows x 3 columns]

""",
        )

    def test6(self):
        s, City, from_city, to_city, volume = initialise()
        t1 = s.get(FromCity=City, ToCity=City)
        t2 = t1.infer(["FromCity", "ToCity"], volume)
        t3 = t2.mutate(volume_fillna=t2["volume"].extend(0))
        t4 = t3.swap("FromCity", "ToCity")
        t5 = t4.shift_left()
        t6 = t5.mutate(total_inflow=t5["volume_fillna"].sum())
        self.maxDiff = None
        self.assertExpectedInline(
            str(t6),
            """\
[ToCity || FromCity volume volume_fillna total_inflow]
                                         FromCity  ... total_inflow
ToCity                                             ...             
London     [Cambridge, Oxford, Edinburgh, London]  ...          6.6
Edinburgh  [Cambridge, Oxford, Edinburgh, London]  ...          4.2
Oxford     [Cambridge, Oxford, Edinburgh, London]  ...          3.0
Cambridge  [Cambridge, Oxford, Edinburgh, London]  ...          6.0

[4 rows x 4 columns]

""",
        )

    def test7(self):
        s, City, from_city, to_city, volume = initialise()
        t1 = s.get(FromCity=City, ToCity=City)
        t2 = t1.infer(["FromCity", "ToCity"], volume)
        t3 = t2.mutate(volume_fillna=t2["volume"].extend(0))
        t4 = t3.swap("FromCity", "ToCity")
        t5 = t4.shift_left()
        t6 = t5.mutate(total_inflow=t5["volume_fillna"].sum())
        t7 = t6.mutate(relative_inflow=t6["volume_fillna"] / t6["total_inflow"])
        self.maxDiff = None
        self.assertExpectedInline(
            str(t7),
            """\
[ToCity || FromCity volume volume_fillna total_inflow relative_inflow]
                                         FromCity  ...                                    relative_inflow
ToCity                                             ...                                                   
London     [Cambridge, Oxford, Edinburgh, London]  ...  [0.4545454545454546, 0.18181818181818182, 0.36...
Edinburgh  [Cambridge, Oxford, Edinburgh, London]  ...  [0.5714285714285714, 0.42857142857142855, 0.0,...
Oxford     [Cambridge, Oxford, Edinburgh, London]  ...  [0.19999999999999998, 0.0, 0.0, 0.799999999999...
Cambridge  [Cambridge, Oxford, Edinburgh, London]  ...                [0.0, 0.0, 0.3, 0.7000000000000001]

[4 rows x 5 columns]

""",
        )

    def test8(self):
        s, City, from_city, to_city, volume = initialise()
        t1 = s.get(FromCity=City, ToCity=City)
        t2 = t1.infer(["FromCity", "ToCity"], volume)
        t3 = t2.mutate(volume_fillna=t2["volume"].extend(0))
        t4 = t3.swap("FromCity", "ToCity")
        t5 = t4.shift_left()
        t6 = t5.mutate(total_inflow=t5["volume_fillna"].sum())
        t7 = t6.mutate(relative_inflow=t6["volume_fillna"] / t6["total_inflow"])
        t8 = t2.infer(["ToCity"], t7["relative_inflow"])
        # TODO
        self.assertExpectedInline(
            str(t8),
            """\
[FromCity ToCity || volume relative_inflow]
                     volume                                    relative_inflow
FromCity  ToCity                                                              
Cambridge London        3.0  [0.4545454545454546, 0.18181818181818182, 0.36...
          Edinburgh     2.4  [0.5714285714285714, 0.42857142857142855, 0.0,...
          Oxford        0.6  [0.19999999999999998, 0.0, 0.0, 0.799999999999...
London    Cambridge     4.2                [0.0, 0.0, 0.3, 0.7000000000000001]
          Oxford        2.4  [0.19999999999999998, 0.0, 0.0, 0.799999999999...
Oxford    London        1.2  [0.4545454545454546, 0.18181818181818182, 0.36...
          Edinburgh     1.8  [0.5714285714285714, 0.42857142857142855, 0.0,...
Edinburgh Cambridge     1.8                [0.0, 0.0, 0.3, 0.7000000000000001]
          London        2.4  [0.4545454545454546, 0.18181818181818182, 0.36...
London    London        NaN  [0.4545454545454546, 0.18181818181818182, 0.36...
          Edinburgh     NaN  [0.5714285714285714, 0.42857142857142855, 0.0,...
Edinburgh Edinburgh     NaN  [0.5714285714285714, 0.42857142857142855, 0.0,...
          Oxford        NaN  [0.19999999999999998, 0.0, 0.0, 0.799999999999...
Oxford    Oxford        NaN  [0.19999999999999998, 0.0, 0.0, 0.799999999999...
          Cambridge     NaN                [0.0, 0.0, 0.3, 0.7000000000000001]
Cambridge Cambridge     NaN                [0.0, 0.0, 0.3, 0.7000000000000001]

""",
        )

    def test9(self):
        s, City, from_city, to_city, volume = initialise()
        t1 = s.get(FromCity=City, ToCity=City)
        t2 = t1.infer(["FromCity", "ToCity"], volume)
        t3 = t2.mutate(volume_fillna=t2["volume"].extend(0))
        t9 = t3.shift_left()
        self.assertExpectedInline(
            str(t9),
            """\
[FromCity || ToCity volume volume_fillna]
                                           ToCity  ...         volume_fillna
FromCity                                           ...                      
Cambridge  [London, Edinburgh, Oxford, Cambridge]  ...  [3.0, 2.4, 0.6, 0.0]
Oxford     [London, Edinburgh, Oxford, Cambridge]  ...  [1.2, 1.8, 0.0, 0.0]
Edinburgh  [London, Edinburgh, Oxford, Cambridge]  ...  [2.4, 0.0, 0.0, 1.8]
London     [London, Edinburgh, Oxford, Cambridge]  ...  [0.0, 0.0, 2.4, 4.2]

[4 rows x 3 columns]

""",
        )

    def test10(self):
        s, City, from_city, to_city, volume = initialise()
        t1 = s.get(FromCity=City, ToCity=City)
        t2 = t1.infer(["FromCity", "ToCity"], volume)
        t3 = t2.mutate(volume_fillna=t2["volume"].extend(0))
        t9 = t3.shift_left()
        t10 = t9.mutate(total_outflow=t9["volume_fillna"].sum())
        self.maxDiff = None
        self.assertExpectedInline(
            str(t10),
            """\
[FromCity || ToCity volume volume_fillna total_outflow]
                                           ToCity  ... total_outflow
FromCity                                           ...              
Cambridge  [London, Edinburgh, Oxford, Cambridge]  ...           6.0
Oxford     [London, Edinburgh, Oxford, Cambridge]  ...           3.0
Edinburgh  [London, Edinburgh, Oxford, Cambridge]  ...           4.2
London     [London, Edinburgh, Oxford, Cambridge]  ...           6.6

[4 rows x 4 columns]

""",
        )

    def test11(self):
        s, City, from_city, to_city, volume = initialise()
        t1 = s.get(FromCity=City, ToCity=City)
        t2 = t1.infer(["FromCity", "ToCity"], volume)
        t3 = t2.mutate(volume_fillna=t2["volume"].extend(0))
        t9 = t3.shift_left()
        t10 = t9.mutate(total_outflow=t9["volume_fillna"].sum())
        t11 = t10.mutate(relative_outflow=t10["volume_fillna"] / t10["total_outflow"])
        self.assertExpectedInline(
            str(t11),
            """\
[FromCity || ToCity volume volume_fillna total_outflow relative_outflow]
                                           ToCity  ...                                   relative_outflow
FromCity                                           ...                                                   
Cambridge  [London, Edinburgh, Oxford, Cambridge]  ...  [0.5, 0.39999999999999997, 0.09999999999999999...
Oxford     [London, Edinburgh, Oxford, Cambridge]  ...               [0.39999999999999997, 0.6, 0.0, 0.0]
Edinburgh  [London, Edinburgh, Oxford, Cambridge]  ...  [0.5714285714285714, 0.0, 0.0, 0.4285714285714...
London     [London, Edinburgh, Oxford, Cambridge]  ...  [0.0, 0.0, 0.36363636363636365, 0.636363636363...

[4 rows x 5 columns]

""",
        )

    def test12(self):
        s, City, from_city, to_city, volume = initialise()
        t1 = s.get(FromCity=City, ToCity=City)
        t2 = t1.infer(["FromCity", "ToCity"], volume)
        t3 = t2.mutate(volume_fillna=t2["volume"].extend(0))
        t4 = t3.swap("FromCity", "ToCity")
        t5 = t4.shift_left()
        t6 = t5.mutate(total_inflow=t5["volume_fillna"].sum())
        t7 = t6.mutate(relative_inflow=t6["volume_fillna"] / t6["total_inflow"])

        t9 = t3.shift_left()
        t10 = t9.mutate(total_outflow=t9["volume_fillna"].sum())
        t11 = t10.mutate(relative_outflow=t10["volume_fillna"] / t10["total_outflow"])

        self.maxDiff = None

        t12 = t1.infer(["ToCity"], t7["relative_inflow"], with_name="expected_outflow")
        print(t12.derivation)
        self.assertExpectedInline(
            str(t12),
            """\
[FromCity ToCity || expected_outflow]
                                                      expected_outflow
FromCity  ToCity                                                      
London    London     [0.4545454545454546, 0.18181818181818182, 0.36...
Edinburgh London     [0.4545454545454546, 0.18181818181818182, 0.36...
Oxford    London     [0.4545454545454546, 0.18181818181818182, 0.36...
Cambridge London     [0.4545454545454546, 0.18181818181818182, 0.36...
London    Edinburgh  [0.5714285714285714, 0.42857142857142855, 0.0,...
Edinburgh Edinburgh  [0.5714285714285714, 0.42857142857142855, 0.0,...
Oxford    Edinburgh  [0.5714285714285714, 0.42857142857142855, 0.0,...
Cambridge Edinburgh  [0.5714285714285714, 0.42857142857142855, 0.0,...
London    Oxford     [0.19999999999999998, 0.0, 0.0, 0.799999999999...
Edinburgh Oxford     [0.19999999999999998, 0.0, 0.0, 0.799999999999...
Oxford    Oxford     [0.19999999999999998, 0.0, 0.0, 0.799999999999...
Cambridge Oxford     [0.19999999999999998, 0.0, 0.0, 0.799999999999...
London    Cambridge                [0.0, 0.0, 0.3, 0.7000000000000001]
Edinburgh Cambridge                [0.0, 0.0, 0.3, 0.7000000000000001]
Oxford    Cambridge                [0.0, 0.0, 0.3, 0.7000000000000001]
Cambridge Cambridge                [0.0, 0.0, 0.3, 0.7000000000000001]

""",
        )

    def test13(self):
        s, City, from_city, to_city, volume = initialise()
        t1 = s.get(FromCity=City, ToCity=City)
        t2 = t1.infer(["FromCity", "ToCity"], volume)
        t3 = t2.mutate(volume_fillna=t2["volume"].extend(0))
        t4 = t3.swap("FromCity", "ToCity")
        t5 = t4.shift_left()
        t6 = t5.mutate(total_inflow=t5["volume_fillna"].sum())
        t7 = t6.mutate(relative_inflow=t6["volume_fillna"] / t6["total_inflow"])

        t9 = t3.shift_left()
        t10 = t9.mutate(total_outflow=t9["volume_fillna"].sum())
        t11 = t10.mutate(relative_outflow=t10["volume_fillna"] / t10["total_outflow"])

        t12 = t1.infer(["ToCity"], t7["relative_inflow"], with_name="expected_outflow")
        t13 = t12.show("FromCity_1").equate("FromCity", "FromCity_1")
        print(t13["expected_outflow"].get_hidden_keys())
        self.maxDiff = None

        self.assertExpectedInline(
            str(t13),
            """\
[FromCity ToCity || expected_outflow]
                     expected_outflow
FromCity  ToCity                     
Cambridge London             0.454545
          Edinburgh          0.571429
          Oxford             0.200000
          Cambridge          0.000000
Oxford    Cambridge          0.000000
          London             0.181818
          Edinburgh          0.428571
          Oxford             0.000000
Edinburgh Oxford             0.000000
          London             0.363636
London    Edinburgh          0.000000
Edinburgh Edinburgh          0.000000
          Cambridge          0.300000
London    London             0.000000
          Oxford             0.800000
          Cambridge          0.700000

""",
        )

    def test14(self):
        s, City, from_city, to_city, volume = initialise()

        t1 = s.get(FromCity=City, ToCity=City)
        t2 = t1.infer(["FromCity", "ToCity"], volume)

        t3 = t2.mutate(volume_fillna=t2["volume"].extend(0))
        t4 = t3.swap("FromCity", "ToCity")
        t5 = t4.shift_left()
        t6 = t5.mutate(relative_inflow=t5["volume_fillna"] / t5["volume_fillna"].sum())

        t7 = t3.shift_left()
        t8 = t7.mutate(relative_outflow=t7["volume_fillna"] / t7["volume_fillna"].sum())

        t9 = (
            t1.infer(["ToCity"], t6["relative_inflow"], with_name="expected_outflow")
            .show("FromCity_1")
            .equate("FromCity", "FromCity_1")
        )

        print("table 9")
        print(t9["expected_outflow"].node.intermediate_representation)

        t10 = (
            t9.infer(["FromCity"], t8["relative_outflow"])
            .show("ToCity_1")
            .equate("ToCity", "ToCity_1")
        )

        self.maxDiff = None

        self.assertExpectedInline(
            str(t10),
            """\
[FromCity ToCity || expected_outflow relative_outflow]
                     expected_outflow  relative_outflow
FromCity  ToCity                                       
Cambridge London             0.454545          0.500000
          Edinburgh          0.571429          0.400000
          Oxford             0.200000          0.100000
Oxford    Cambridge          0.000000          0.000000
Cambridge Cambridge          0.000000          0.000000
London    Cambridge          0.700000          0.636364
          Oxford             0.800000          0.363636
          London             0.000000          0.000000
          Edinburgh          0.000000          0.000000
Edinburgh Edinburgh          0.000000          0.000000
Oxford    London             0.181818          0.400000
          Edinburgh          0.428571          0.600000
Edinburgh Oxford             0.000000          0.000000
Oxford    Oxford             0.000000          0.000000
Edinburgh Cambridge          0.300000          0.428571
          London             0.363636          0.571429

""",
        )
