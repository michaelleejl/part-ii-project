class Table:
    def __init__(self, keys, values, derivation, schema):
        self.keys = keys
        self.values = values
        self.derivation = derivation
        self.schema = schema

    def compose(self, with_edge):
        pass

    def infer(self, fr, to):
        pass

    def combine(self, with_table):
        pass

    def hide(self, key):
        pass

    def show(self, key):
        pass

    def make_value(self, node):
        pass

    def __repr__(self):
        return f"[{' '.join([str(k) for k in self.keys])} || {' '.join([str(v) for v in self.values])}]"

    def __str__(self):
        return self.__repr__()


    ## the task is
    ## given a schema where (bank | cardnum) and (bonus | cardnum, person)
    ## I want [cardnum person || bank bonus] as the values

    ## t = schema.get([cardnum, person]) [cardnum person || unit]
    ## 2 possibilities: cardnum x person or the specific cardnum, person pairs that key bonus.

    ## t = t.infer([cardnum, person] -> cardnum)
    ## t = t.infer(cardnum -> bank).add_value(bank)
    ## t = t.infer([cardnum, person] -> bonus).add_value(bonus)



    ## Example of composition
    ## Schema: Order -> payment method -> billing address
    ## Goal is: [order || billing address]

    ## I can do
    ## t = schema.get([payment method]) [payment method || unit]

    ## t = t.compose(order -> payment method)
    ## t = t.infer(payment method -> billing address).add_value(billing address) [order || billing address]
    ## t = t.infer(billing address -> shipping fee)

    ## Order
    ##  |
    ## payment method
    ##  |
    ## billing address

    ## turn them into test suites
