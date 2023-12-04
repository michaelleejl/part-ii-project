from schema import SchemaNode, SchemaEdge, Cardinality


class FullyConnected:
    def __init__(self,
                 nodes: frozenset[SchemaNode],
                 key_node: SchemaNode,
                 cardinalities: dict[SchemaEdge, Cardinality] = None):
        self.nodes = nodes
        self.key_node = key_node
        if cardinalities is None:
            self.cardinalities = {}
        else:
            self.cardinalities = cardinalities

    def get_edge(self, from_node, to_node):
        from_node_c = from_node.constituents
        to_node_c = to_node.constituents

        assert from_node_c <= self.nodes
        assert to_node_c <= self.nodes

        edge = SchemaEdge(from_node, to_node, Cardinality.MANY_TO_MANY)

        if from_node == self.key_node:
            edge.cardinality = Cardinality.MANY_TO_ONE

        if edge in self.cardinalities:
            cardinality = self.cardinalities[edge]
            edge.cardinality = cardinality

        return edge

    def specify_cardinality(self, from_node, to_node, cardinality):
        edge = SchemaEdge(from_node, to_node, Cardinality.MANY_TO_MANY)
        self.cardinalities[edge] = cardinality
