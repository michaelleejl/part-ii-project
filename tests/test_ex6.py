import expecttest
import pandas as pd

from schema import Schema, SchemaNode


class TestEx6(expecttest.TestCase):
    def test_ex6(self):
        s = Schema()

        billing = pd.read_csv("./csv/orders/billing.csv").set_index("payment_method")
        payment = pd.read_csv("./csv/orders/payment.csv").set_index("order")
        delivery = pd.read_csv("./csv/orders/delivery.csv").set_index("order")

        s.insert_dataframe(billing, "billing")
        s.insert_dataframe(billing, "payment")
        s.insert_dataframe(billing, "delivery")

        p_order = SchemaNode("order", cluster="payment")
        d_order = SchemaNode("order", cluster="delivery")

        p_payment_method = SchemaNode("payment_method", cluster="payment")
        b_payment_method = SchemaNode("payment_method", cluster="billing")

        b_billing_address = SchemaNode("billing_address", cluster="billing")
        d_delivery_address = SchemaNode("delivery_address", cluster="delivery")

        s.blend(p_order, d_order, as ="Order")
        s.blend(p_payment_method, b_payment_method, as="Payment_Method")
        s.blend(d_delivery_address, b_billing_address, as="Address")

        # ========================================================================
        # ========================================================================

        # SCHEMA:
        # order         ---> delivery_address
        #   |                       |
        #   |                       |
        #   v                       |
        # payment_method ---> billing_address
        #
        # GOAL: [order payment_method || address]

        t1 = s.get([p_order, p_payment_method])
        # [payment.order payment.payment_method || ]
        #  1             5172
        #  1             2354
        #  1             1410
        #  1             1410
        #  ...

        t2 = t1.infer(["payment.order"], SchemaNode("Address"))
        # [payment.order payment.payment_method || Address ]
        #  1             5172                   || Cambridge
        #  1             2354                   || Cambridge
        #  1             1410                   || Cambridge
        #  1             1410                   || Cambridge

        # This is the same as Ex5

        # STRESS TEST
        t11 = s.get([p_order]).infer(["payment.order"], p_payment_method)
        # [payment.order || payment.payment_method]
        #  1             || 5172
        #  2             || 2354
        #  4             || 1410
        #  5             || 1111

        t12 = t11.set_key(["payment.order", "payment.method"])
        # [payment.order   payment.payment_method  || ]
        #  1               5172
        #  2               2354
        #  4               1410
        #  5               1111

        # This is not the same as s.get([p_order, p_payment_method])
        # Nor should it be! There is a relationship in t11 that should be preserved
        # But I'm curious to hear if you think this is expected or surprising