import copy

from tables.internal_representation import *


def flatten(xss):
    return [x for xs in xss for x in xs]

class NodeIntermediateRepresentation:
    def __init__(self, prefix, main, suffix):
        self.prefix = copy.copy(prefix) #Prefix: how to get from parent node to start node.
        self.main = copy.copy(main)     #Output from the schema graph. How to get from start node to end node
        self.suffix = copy.copy(suffix) #Things you need to do after the end node

    def compose(self, to_push):
        return NodeIntermediateRepresentation(
            [to_push, Call()] + self.prefix,
            self.main,
            self.suffix + [Return()]
        )

    def strip(self):
        return NodeIntermediateRepresentation(
            self.prefix[1:],
            self.main,
            self.suffix[:-1]
        )

    def get_prefix(self):
        return flatten(self.prefix)

    def get_suffix(self):
        return flatten(self.suffix)

    def get_main(self):
        return self.main

    def generate(self):
        return self.get_prefix() + self.get_main() + self.get_suffix()
