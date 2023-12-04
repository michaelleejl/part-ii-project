class DerivationNode:
    def __init__(self, node, children):
        self.node = node
        self.children = children

    def add_child(self, child):
        self.children = self.children.union([child])