from schema import SchemaNode, SchemaEdge, Cardinality


class FullyConnected:
    def __init__(self,
                 nodes: list[SchemaNode],
                 key_node: SchemaNode):
        self.nodes = nodes
        self.key_node = key_node

    def get_edge(self, from_node: SchemaNode, to_node: SchemaNode):
        from_node_c = SchemaNode.get_constituents(from_node)
        to_node_c = SchemaNode.get_constituents(to_node)

        assert frozenset(from_node_c) <= frozenset(self.nodes)
        assert frozenset(to_node_c) <= frozenset(self.nodes)

        edge = SchemaEdge(from_node, to_node)

        if from_node == self.key_node:
            edge.cardinality = Cardinality.MANY_TO_ONE
        if to_node == self.key_node:
            edge.cardinality = Cardinality.ONE_TO_MANY

        return edge

    def __contains__(self, node):
        constituents = SchemaNode.get_constituents(node)
        return constituents.issubset(self.nodes)

    def __repr__(self):
        return "\n".join(list(map(str, self.nodes)))


