from schema import SchemaNode, SchemaEdge, Schema


class Key:
    def __init__(self, node: SchemaNode, schema: Schema):
        self.schema = schema
        self.node = node

    def compose(self, edge: SchemaEdge, node: SchemaNode = None):
        node = edge.traverse(self.node)
        edges = self.schema.get_edges_attached_to_node(node)
