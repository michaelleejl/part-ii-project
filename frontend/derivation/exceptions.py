from frontend.domain import Domain


class KeysMustBeUniqueException(Exception):
    def __init__(self, keys: list[Domain]):
        super().__init__(f"Keys {keys} are not unique")


class NodeIsAlreadyChildOfParentException(Exception):
    def __init__(self, node, parent):
        child_str = ";".join([str(d) for d in node.domains])
        parent_str = ";".join([str(d) for d in parent.domains])
        super().__init__(f"Node {child_str} is already a child of {parent_str}")


class NodeIsNotChildOfParentException(Exception):
    def __init__(self, node, parent):
        child_str = ";".join([str(d) for d in node.domains])
        parent_str = ";".join([str(d) for d in parent.domains])
        super().__init__(f"Node {child_str} is not a child of {parent_str}")


class PathMustNotBeEmptyException(Exception):
    def __init__(self):
        super().__init__(f"Path can not be empty")


class KeysNotFoundException(Exception):
    def __init__(self, keys: list[Domain]):
        super().__init__(f"Keys {keys} not found in derivation tree")


class ValuesNotFoundException(Exception):
    def __init__(self, values: list[Domain]):
        super().__init__(f"Keys {values} not found in derivation tree")


class PathNotFoundException(Exception):
    def __init__(self, keys: list[Domain], values: list[Domain]):
        super().__init__(f"No path from {keys} to {values} found in derivation tree")


class PathShouldDivergeException(Exception):
    def __init__(self):
        super().__init__(f"Path should diverge")