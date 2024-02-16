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
