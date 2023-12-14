def is_reachable_via_projection(from_node, to_node):
    from schema.node import SchemaNode
    return set(SchemaNode.get_constituents(from_node)) > set(SchemaNode.get_constituents(to_node))