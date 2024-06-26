class ClusterAlreadyExistsException(Exception):
    def __init__(self, family):
        super().__init__(f"Family {family} already exists. Use `update` instead.")


class ExtendMappingShouldConflictWithOldMappingException(Exception):
    def __init__(self, vals):
        super().__init__(
            f"Values {vals} already exist in the old mapping. Use `update` instead."
        )


class EdgeAlreadyExistsException(Exception):
    def __init__(self, edge):
        super().__init__(
            f"Edge between {edge.from_node} and {edge.to_node} already exists. Use `replace` instead."
        )


class EdgeDoesNotExistException(Exception):
    def __init__(self, edge):
        super().__init__(f"Edge {edge} does not exist. Use `add` instead.")


class NodeNotInSchemaGraphException(Exception):
    def __init__(self, node):
        super().__init__(f"Node {node} is not in schema graph.")


class NodeAlreadyInSchemaGraphException(Exception):
    def __init__(self, node):
        super().__init__(f"Node {node} is already in schema graph.")


class EdgeIsNotConnectedToNodeException(Exception):
    def __init__(self, edge, node):
        super().__init__(f"Node {node} is not connected to edge {edge}")


class EdgeDoesNotExistBetweenNodesException(Exception):
    def __init__(self, node1, node2):
        super().__init__(f"{node1} is not connected to {node2} by an edge")


class NodesDoNotExistInGraphException(Exception):
    def __init__(self, nodes):
        super().__init__(f"{nodes} do not exist in graph")


class TableShouldNotHaveDuplicateKeysException(Exception):
    def __init__(self):
        super().__init__("Table should not have duplicate keys")


class AllNodesInClusterMustAlreadyBeInGraphException(Exception):
    def __init__(self, nodes):
        super().__init__(
            "When adding a cluster, all nodes in the cluster must already exist in the graph. "
            + f"The following nodes are not in the graph: {','.join(list(map(str, nodes)))}"
        )


class AllNodesInFullyConnectedClusterMustHaveSameClusterException(Exception):
    def __init__(self):
        super().__init__(
            "All nodes in a fully connected cluster must have the same cluster attribute"
        )


class FindingEdgeViaNodeMustRespectEquivalence(Exception):
    def __init__(self, node1, via):
        super().__init__(
            f"When finding an edge between node1 and node2 via node3, node1 and node3 must be equivalent. "
            f"{node1} and {via} are not equivalent."
        )


class NoShortestPathBetweenNodesException(Exception):
    def __init__(self, node1, node2):
        super().__init__(
            f"No paths found between nodes {node1} and {node2}. "
            f"If the path involves a projection, the projection will need to be specified as a waypoint."
        )


class MultipleShortestPathsBetweenNodesException(Exception):
    def __init__(self, node1, node2, shortest_paths):
        super().__init__(
            f"Multiple shortest paths found between nodes {node1} and {node2}. "
            f"Shortest paths: {sorted(shortest_paths, key=lambda xs: ' '.join([str(x) for x in xs]))}"
        )


class CycleDetectedInPathException(Exception):
    def __init__(self):
        super().__init__(f"Cycle detected in path.")


class ClassAlreadyExistsException(Exception):
    def __init__(self):
        super().__init__(f"Class already exists.")


class CannotRenameClassException(Exception):
    def __init__(self):
        super().__init__(f"Class cannot be renamed.")


class CannotInsertDataFrameIfSchemaBackedBySQLBackendException(Exception):
    def __init__(self):
        super().__init__(
            "Cannot insert dataframe if schema is backed by non-pandas backend"
        )


class SchemaClassMustBeSpecifiedException(Exception):
    def __init__(self):
        super().__init__(
            "Both nodes have no associated class. SchemaClass must be specified"
        )


class CannotBlendNodesUnderDifferentClassesException(Exception):
    def __init__(self):
        super().__init__("Cannot blend nodes under different classes")


class CannotBlendNodesWithDifferentTypeException(Exception):
    def __init__(self, node1, node2):
        super().__init__(
            f"Cannot blend nodes with different type. Nodes have type {node1.node_type} and {node2.node_type}"
        )


class ColumnMustBeAnAtomicNodeOrClassException(Exception):
    def __init__(self, name):
        super().__init__(f"Column {name} must be an atomic node or class")
