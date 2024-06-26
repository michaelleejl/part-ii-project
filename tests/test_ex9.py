import expecttest
import pandas as pd

from schema.schema import Schema


class TestEx9(expecttest.TestCase):

    def initialise(self):
        s = Schema()
        kl_df = pd.read_csv("./csv/abstract/l.csv").set_index("k")
        lv_df = pd.read_csv("./csv/abstract/v.csv").set_index("l")
        kl = s.insert_dataframe(kl_df)
        lv = s.insert_dataframe(lv_df)
        L = s.create_class("L")
        s.blend(kl["l"], lv["l"], L)
        return s, kl, lv, L
        # [k || l]
        #  A || p
        #  B || p
        #  C || q

        # [l || v]
        #  p || 1
        #  q || 2
        #  r || 3

    def test_ex9_goal1_step1_get_infer_compose(self):
        # GOAL 1: [k l || v]
        s, kl, lv, L = self.initialise()
        t1 = s.get(k=kl["k"], l=kl["l"]).infer(["k"], lv["v"])
        self.assertExpectedInline(
            str(t1),
            """\
[k l || v]
     v
k l   
A p  1
B p  1
C p  2
A q  1
B q  1
C q  2

""",
        )
        # Or s.get([l]).compose(k, [l]).infer([k], v)
        # [k || v]
        #  A || 1
        #  B || 1
        #  C || 2

    # GOAL 2 [k l || v]

    def test_ex9_goal2_step1_getAndInfer(self):
        s, kl, lv, L = self.initialise()
        t2 = s.get(k=kl["k"], l=kl["l"]).infer(["l"], lv["v"])

        self.assertExpectedInline(
            str(t2),
            """\
[k l || v]
     v
k l   
A p  1
B p  1
C p  1
A q  2
B q  2
C q  2

""",
        )
        # [k  l || v]
        #  A  p || 1
        #  A  q || 2
        #  A  r || 3
        #  B  p || 1
        #  B  q || 2
        #  B  r || 3
        #  C  p || 1
        #  C  q || 2
        #  C  r || 3

    def test_ex9_goal3_step1_getAndInfer(self):
        s, kl, lv, L = self.initialise()
        t21 = s.get(k=kl["k"], L=L).infer(["k"], lv["v"]).sort(["k", "L"])
        self.assertExpectedInline(
            str(t21),
            """\
[k L || v]
     v
k L   
A p  1
  q  1
  r  1
B p  1
  q  1
  r  1
C p  2
  q  2
  r  2

""",
        )
        # [k  l || v]
        #  A  p || 1
        #  A  q || 1
        #  A  r || 1
        #  B  p || 1
        #  B  q || 1
        #  B  r || 1
        #  C  p || 2
        #  C  q || 2
        #  C  r || 2

    def test_ex9_goal4_step1_getAndInfer(self):
        s, kl, lv, L = self.initialise()
        t31 = s.get(k=kl["k"]).infer(["k"], L)
        self.assertExpectedInline(
            str(t31),
            """\
[k || L]
   L
k   
A  p
B  p
C  q

""",
        )
        # [k || l]
        #  A || p
        #  B || p
        #  C || q

    def test_ex9_goal4_step2_setKey(self):
        s, kl, lv, L = self.initialise()
        t31 = s.get(k=kl["k"]).infer(["k"], L)
        t32 = t31.shift_right()
        self.assertExpectedInline(
            str(t32),
            """\
[k L || ]
Empty DataFrame
Columns: []
Index: []
9 keys hidden

""",
        )
        # [k l || ]
        #  A p
        #  B p
        #  C q

    def test_ex9_goal4_step3_infer(self):
        s, kl, lv, L = self.initialise()
        t31 = s.get(k=kl["k"]).infer(["k"], L)
        t32 = t31.shift_right()
        t33 = t32.infer(["L"], lv["v"])
        self.assertExpectedInline(
            str(t33),
            """\
[k L || v]
     v
k L   
A p  1
B p  1
C q  2
6 keys hidden

""",
        )
        # [k l || v ]
        #  A p || 1
        #  B p || 1
        #  C q || 2

        # WAY 4 (STRESS TEST)
        # t41 = s.get(["1.k", "1.l"]).infer(["1.k", "1.l"], "2.v")
        # Should throw an exception!
        # Any inference paths that involve a projection
        # Require the user to be explicit about what that projection is
        # Because it corresponds to a choice amongst derivations
        # This overrides any shortest path heuristics.

        # Note that these methods are semantically different!
        # But I think the syntax forces the user to be clear about what they want
