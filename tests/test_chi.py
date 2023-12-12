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
        t1 = s.get(["City", "City"], ["FromCity", "ToCity"])
        print(t1)
        t2 = t1.infer(["FromCity", "ToCity"], "flows.volume")
        print(t2)
        fill_na = lambda x: 0 if pd.isnull(x) else x
        t3 = t2.assign("volume", t2["flows.volume"].apply(fill_na, Cardinality.ONE_TO_ONE)).sort(
             ["FromCity", "ToCity"]).hide("flows.volume")
        print(t3)

        ## compute total inflow
        t4 = t3.set_key(["ToCity"])
        print(t4)
        t5 = t4.assign("total_inflow", t4["volume"].aggregate(sum)).sort(["FromCity", "ToCity"])
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
        # TODO: Define aggregation over columns with no hidden keys
        print(t9)

        t10 = (t9.assign("relative_outflow", t9["volume"] / t9["total_outflow"])
                 .assign("expected_outflow", t9["total_inflow"] / 21)
                 .set_key(["FromCity"]))
        print(t10)
        #
        # def chi(x, y):
        #     from scipy.stats import chisquare
        #     print(x)
        #     print(y)
        #     return chisquare(np.array(x), np.array(y)).pvalue
        # chi_square = create_function(chi)
        #
        # def pair(x, y):
        #     return [x, y]
        #
        # t11 = t10.assign("relative_vs_expected", t10["relative_outflow"] / t10["expected_outflow"])
        # print(t11)



    # print(t5)


