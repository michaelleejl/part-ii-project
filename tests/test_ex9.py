import expecttest
import pandas as pd

from schema import Schema, SchemaNode


class TestEx8(expecttest.TestCase):
    def test_ex_9(self):
        s = Schema()

        # [k || l]
        #  A || p
        #  B || p
        #  C || q

        # [l || v]
        #  p || 1
        #  q || 2
        #  r || 3

        # ========================================================================
        # ========================================================================

        # GOAL 1: [k || v]
        k = SchemaNode("k")
        l = SchemaNode("l")
        v = SchemaNode("v")
        t1 = s.get([l]).infer([l], v).compose(k, [l])
        # Or s.get([l]).compose(k, [l]).infer([k], v)
        # [k || v]
        #  A || 1
        #  B || 1
        #  C || 2

        # GOAL 2 [k l || v]

        # WAY 1
        t11 = s.get([k, l]).infer([l], v)
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

        # WAY 2
        t21 = s.get([k, l]).infer([k], v)
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

        # WAY 3
        t31 = s.get([k]).infer([k], l)
        # [k || l]
        #  A || p
        #  B || p
        #  C || q

        t32 = t31.set_key([k, l])
        # [k l || ]
        #  A p
        #  B p
        #  C q

        t33 = t32.infer([l], v)
        # [k l || v ]
        #  A p || 1
        #  B p || 1
        #  C q || 2

        # WAY 4 (STRESS TEST)
        t41 = s.get([k, l]).infer([k, l], v)
        # Should throw an exception!
        # Any inference paths that involve a projection
        # Require the user to be explicit about what that projection is
        # Because it corresponds to a choice amongst derivations
        # This overrides any shortest path heuristics.

        # Note that these methods are semantically different!
        # But I think the syntax forces the user to be clear about what they want
