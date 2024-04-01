import expecttest
import pandas as pd

from schema.schema import Schema


class TestEx6(expecttest.TestCase):

    def initialise(self):
        s = Schema()

        billing_df = pd.read_csv("./csv/orders/billing.csv").set_index("payment_method")
        payment_df = pd.read_csv("./csv/orders/payment.csv").set_index("order")
        delivery_df = pd.read_csv("./csv/orders/delivery.csv").set_index("order")

        billing = s.insert_dataframe(billing_df)
        payment = s.insert_dataframe(payment_df)
        delivery = s.insert_dataframe(delivery_df)

        p_order = payment["order"]
        d_order = delivery["order"]

        p_payment_method = payment["payment_method"]
        b_payment_method = billing["payment_method"]

        b_billing_address = billing["billing_address"]
        d_delivery_address = delivery["delivery_address"]

        Order = s.create_class("Order")
        Payment_Method = s.create_class("Payment_Method")
        Address = s.create_class("Address")

        s.blend(p_order, d_order, under=Order)
        s.blend(p_payment_method, b_payment_method, under=Payment_Method)
        s.blend(d_delivery_address, b_billing_address, under=Address)
        return s, billing, delivery, payment, Address, Order, Payment_Method

        # SCHEMA:
        # order         ---> delivery_address
        #   |                       |
        #   |                       |
        #   v                       |
        # payment_method ---> billing_address

    # GOAL: [order payment_method || address]

    def test_ex6_goal1_step1_get(self):
        s, billing, delivery, payment, Address, Order, Payment_Method = (
            self.initialise()
        )
        t1 = s.get(order=payment["order"], payment_method=payment["payment_method"])
        self.assertExpectedInline(
            str(t1),
            """\
[order payment_method || ]
Empty DataFrame
Columns: []
Index: []
16 keys hidden

""",
        )
        # [payment.order payment.payment_method || ]
        #  1             5172
        #  1             2354
        #  1             1410
        #  1             1410
        #  ...

    def test_ex6_goal1_step2_infer(self):
        s, billing, delivery, payment, Address, Order, Payment_Method = (
            self.initialise()
        )
        t1 = s.get(order=payment["order"], payment_method=payment["payment_method"])
        t2 = t1.infer(["order"], Address).sort(["order", "payment_method"])
        self.assertExpectedInline(
            str(t2),
            """\
[order payment_method || Address]
                        Address
order payment_method           
1.0   1111            Cambridge
      1410            Cambridge
      2354            Cambridge
      5172            Cambridge
2.0   1111            Singapore
      1410            Singapore
      2354            Singapore
      5172            Singapore
4.0   1111               Oxford
      1410               Oxford
      2354               Oxford
      5172               Oxford
5.0   1111            Cambridge
      1410            Cambridge
      2354            Cambridge
      5172            Cambridge

""",
        )
        # [payment.order payment.payment_method || Address ]
        #  1             5172                   || Cambridge
        #  1             2354                   || Cambridge
        #  1             1410                   || Cambridge
        #  1             1410                   || Cambridge

        # This is the same as Ex5

    def test_ex6_goal2_infer(self):
        s, billing, delivery, payment, Address, Order, Payment_Method = (
            self.initialise()
        )
        t1 = s.get(order=payment["order"], payment_method=payment["payment_method"])
        t2 = t1.infer(["payment_method"], Address).sort(["order", "payment_method"])
        self.assertExpectedInline(
            str(t2),
            """\
[order payment_method || Address]
                        Address
order payment_method           
1     1111.0          Cambridge
      1410.0             London
      2354.0          Singapore
      5172.0             London
2     1111.0          Cambridge
      1410.0             London
      2354.0          Singapore
      5172.0             London
4     1111.0          Cambridge
      1410.0             London
      2354.0          Singapore
      5172.0             London
5     1111.0          Cambridge
      1410.0             London
      2354.0          Singapore
      5172.0             London

""")
        
    # STRESS TEST
    def test_ex6_goal3_step1_getAndInfer(self):
        s, billing, delivery, payment, Address, Order, Payment_Method = (
            self.initialise()
        )
        t11 = s.get(order = payment["order"]).infer(["order"], payment["payment_method"])
        self.assertExpectedInline(
            str(t11),
            """\
[order || payment_method]
       payment_method
order                
1                5172
2                2354
4                1410
5                1111

""",
        )
        # [payment.order || payment.payment_method]
        #  1             || 5172
        #  2             || 2354
        #  4             || 1410
        #  5             || 1111

    def test_ex6_goal3_step2_setKey(self):
        s, billing, delivery, payment, Address, Order, Payment_Method = (
            self.initialise()
        )
        t11 = s.get(order = payment["order"]).infer(["order"], payment["payment_method"])
        t12 = t11.shift_right()
        self.assertExpectedInline(
            str(t12),
            """\
[order payment_method || ]
Empty DataFrame
Columns: []
Index: []
16 keys hidden

""",
        )
        # [payment.order   payment.payment_method  || ]
        #  1               5172
        #  2               2354
        #  4               1410
        #  5               1111

        # This is not the same as s.get([p_order, p_payment_method])
        # Nor should it be! There is a relationship in t11 that should be preserved
        # But I'm curious to hear if you think this is expected or surprising
