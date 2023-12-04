from schema import SchemaNode, SchemaEdge, Cardinality


class FullyConnected:
    def __init__(self,
                 nodes: frozenset[SchemaNode],
                 key_node: SchemaNode):
        self.nodes = nodes
        self.key_node = key_node

    def get_edge(self, from_node: SchemaNode, to_node: SchemaNode):
        from_node_c = SchemaNode.get_constituents(from_node)
        to_node_c = SchemaNode.get_constituents(to_node)

        assert from_node_c <= self.nodes
        assert to_node_c <= self.nodes

        edge = SchemaEdge(from_node, to_node)

        if from_node == self.key_node or from_node < to_node:
            edge.cardinality = Cardinality.MANY_TO_ONE
        if from_node == to_node:
            edge.cardinality = Cardinality.ONE_TO_ONE

        return edge

    def __contains__(self, node):
        constituents = SchemaNode.get_constituents(node)
        return constituents.issubset(self.nodes)
