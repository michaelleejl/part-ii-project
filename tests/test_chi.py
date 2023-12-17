import expecttest
import numpy as np
import pandas as pd

from schema import Schema, SchemaNode, Cardinality
from tables.function import create_function


class TestChi(expecttest.TestCase):
    def test(self):
        s = Schema()
        flows = pd.read_csv("./csv/flows/flows.csv").set_index(["from_city", "to_city"])
        s.insert_dataframe(flows, "flows")
        s.blend(SchemaNode("from_city", cluster="flows"), SchemaNode("to_city", cluster="flows"), under="City")

        t0 = s.get(["flows.from_city", "flows.to_city"])
        t01 = t0.infer(["flows.to_city", "flows.from_city"], "flows.volume")
        print(t01)

        t0 = s.get(["City", "City"], ["FromCity", "ToCity"])
        t01 = t0.infer(["ToCity", "FromCity"], "flows.volume")
        print(t01)

        t1 = s.get(["City", "City"], ["FromCity", "ToCity"])
        print(t1)

        t2 = t1.infer(["ToCity", "FromCity"], "flows.volume")
        print(t2)

        fill_na = lambda x: 0 if pd.isnull(x) else x
        t3 = t2.assign("volume", t2["flows.volume"].apply(fill_na, Cardinality.ONE_TO_ONE)).sort(
             ["FromCity", "ToCity"])
        print(t3)

        ## compute total inflow
        t4 = t3.set_key(["ToCity"])
        t4b = t4.assign("total_inflow", t4["volume"].aggregate(sum)).sort(["FromCity", "ToCity"])
        t4a = t4b.hide("volume")
        print(t4a)
        t5 = t4b.assign("total_inflows", t4b["total_inflow"].aggregate(sum))
        print(t5)


        ## compute total outflow
        t6 = t3.set_key(["FromCity"])
        print(t6)
        t7 = t6.assign("total_outflow", t6["volume"].aggregate(sum)).sort(["FromCity", "ToCity"])
        print(t7)



        ## combine them back into the same table
        t8 = t3.infer(["ToCity"], "total_inflow")
        print(t8)
        t9 = t8.infer(["FromCity"], "total_outflow")
        print(t9)


        # t10 = (t9.assign("relative_outflow", t9["volume"] / t9["total_outflow"])
        #          .assign("expected_outflow", t9["total_inflow"] / t9["total_inflows"])
        #          .set_key(["FromCity"]).hide("volume").hide("total_outflow").hide("ToCity").hide("total_inflow").hide("total_inflows"))
        # print(t10)

    # print(t5)


