from schema.node import SchemaNode


class SchemaClass(SchemaNode):
    def __init__(self, name):
        super().__init__(name)