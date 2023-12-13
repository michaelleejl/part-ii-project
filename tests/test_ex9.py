import expecttest
import pandas as pd

from schema import Schema, SchemaNode


class TestEx9(expecttest.TestCase):

    def initialise(self):
        s = Schema()
        l = pd.read_csv("./csv/abstract/l.csv").set_index("k")
        v = pd.read_csv("./csv/abstract/v.csv").set_index("l")
        s.insert_dataframe(l, "1")
        s.insert_dataframe(v, "2")
        s.blend(SchemaNode("l", cluster="1"), SchemaNode("l", cluster="2"), under="L")
        return s
        # [k || l]
        #  A || p
        #  B || p
        #  C || q

        # [l || v]
        #  p || 1
        #  q || 2
        #  r || 3

    def test_ex9_goal1_step1_get_infer_compose(self):
        # GOAL 1: [k || v]
        s = self.initialise()
        t1 = s.get(["L"]).infer(["L"], "2.v").compose(["1.k"], "L")
        self.assertExpectedInline(str(t1), """\
[1.k || 2.v]
     2.v
1.k     
A      1
B      1
C      2
1 values hidden

""")
        # Or s.get([l]).compose(k, [l]).infer([k], v)
        # [k || v]
        #  A || 1
        #  B || 1
        #  C || 2

    # GOAL 2 [k l || v]

    def test_ex9_goal2_step1_getAndInfer(self):
        s = self.initialise()
        t11 = s.get(["1.k", "L"]).infer(["L"], "2.v")
        self.assertExpectedInline(str(t11), """\
[1.k L || 2.v]
       2.v
1.k L     
A   p    1
B   p    1
C   p    1
A   q    2
B   q    2
C   q    2
A   r    3
B   r    3
C   r    3

""")
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
        s = self.initialise()
        t21 = s.get(["1.k", "L"]).infer(["1.k"], "2.v")
        self.assertExpectedInline(str(t21), """\
[1.k L || 2.v]
       2.v
1.k L     
A   p    1
    q    1
    r    1
B   p    1
    q    1
    r    1
C   p    2
    q    2
    r    2
1 values hidden

""")
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
        s = self.initialise()
        t31 = s.get(["1.k"]).infer(["1.k"], "L")
        self.assertExpectedInline(str(t31), """\
[1.k || L]
     L
1.k   
A    p
B    p
C    q

""")
        # [k || l]
        #  A || p
        #  B || p
        #  C || q

    def test_ex9_goal4_step2_setKey(self):
        s = self.initialise()
        t31 = s.get(["1.k"]).infer(["1.k"], "L")
        t32 = t31.set_key(["1.k", "L"])
        self.assertExpectedInline(str(t32), """\
[1.k L || ]
Empty DataFrame
Columns: []
Index: [(A, p), (B, p), (C, q)]

""")
        # [k l || ]
        #  A p
        #  B p
        #  C q

    def test_ex9_goal4_step3_infer(self):
        s = self.initialise()
        t31 = s.get(["1.k"]).infer(["1.k"], "L")
        t32 = t31.set_key(["1.k", "L"])
        t33 = t32.infer(["L"], "2.v")
        self.assertExpectedInline(str(t33), """\
[1.k L || 2.v]
       2.v
1.k L     
A   p    1
B   p    1
C   q    2
1 values hidden

""")
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
