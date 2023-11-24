class KeyDuplicationException(Exception):
    def __init__(self, msg):
        super().__init__(f"Duplicate key/s detected: {msg}.")


class FamilyAlreadyExistsException(Exception):
    def __init__(self, family):
        super().__init__(f"Family {family} already exists. Use `update` instead.")


class ExtendMappingShouldConflictWithOldMapping(Exception):
    def __init__(self, vals):
        super().__init__(f"Values {vals} already exist in the old mapping. Use `update` instead.")


class EdgeAlreadyExistsException(Exception):
    def __init__(self, edge):
        super().__init__(f"Edge {edge} already exists. Use `replace` instead.")


class EdgeDoesNotExistException(Exception):
    def __init__(self, edge):
        super().__init__(f"Edge {edge} does not exist. Use `add` instead.")


class EdgeIsNotConnectedToNodeException(Exception):
    def __init__(self, edge, node):
        super().__init__(f"Node {node} is not connected to edge {edge}")