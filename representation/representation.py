from __future__ import annotations
import abc
from frontend.domain import Domain


class RepresentationStep(abc.ABC):
    """
    An abstract class that represents an expression in the Intermediate Representation
    """

    def __init__(self, name: str):
        """
        Initialise a new representation step with name

        Args:
            name (str): The name of the (type) of representation expression
        """
        self.name = name

    @abc.abstractmethod
    def __repr__(self):
        pass

    @abc.abstractmethod
    def __str__(self):
        pass

    def invert(self) -> RepresentationStep:
        """
        Invert the representation step
        For example, if the representation step is a traversal, the inverted representation step will be the reverse traversal
        """
        return self

    from schema.node import SchemaNode

    def get_hidden_keys(self) -> list[Domain]:
        return []


class StartTraversal(RepresentationStep):
    """
    A representation expression that represents the beginning of a series of traversals
    A command to return a table with the start_columns copied and numbered 0...n
    """

    def __init__(self, start_columns: list[Domain]):
        """
        Initialise a new StartTraversal representation step

        Args:
            start_columns (list[Domain]): The columns used to start the traversal
        """

        super().__init__("STT")
        self.start_columns = start_columns

    def __repr__(self):
        return f"{self.name} <{self.start_columns}>"

    def __str__(self):
        return self.__repr__()


class EndTraversal(RepresentationStep):
    """
    A representation expression that represents the end of a series of traversals
    A command to return a table with all named columns and the end columns renamed from 0...m to the names given
    """

    def __init__(self, end_columns: list[Domain]):
        """
        Initialise a new EndTraversal representation step
        Args:
            end_columns (list[Domain]): The columns at the end of the traversal
        """
        super().__init__("ENT")
        self.end_columns = end_columns

    def __repr__(self):
        return f"{self.name} <{self.end_columns}>"

    def __str__(self):
        return self.__repr__()


class Traverse(RepresentationStep):
    """
    A representation expression that represents a traversal from one node to another
    If the edge traversed is X --> Y, where there are n atomic nodes in X,
    and the table is of the form A, 0, ..., n
    Then return a table of the form A, H, 0, ..., m, where m is the number of atomic nodes in Y
    and H are the hidden keys required to traverse the edge
    """

    from schema.edge import SchemaEdge
    from schema.node import SchemaNode

    def __init__(self, edge: SchemaEdge, columns=None):
        """
        Initialise a new Traverse representation step

        Args:
            edge (SchemaEdge): The edge to traverse
            columns (list[Domain], optional): Transforms the hidden keys from nodes to domains (named nodes).
            Defaults to None.
        """
        super().__init__("TRV")
        self.edge = edge
        self.hidden_keys = [Domain(node.name, node) for node in edge.get_hidden_keys()]
        self.start_node = edge.from_node
        self.end_node = edge.to_node
        if columns is None:
            self.columns = []
        else:
            self.columns = columns

    def get_hidden_keys(self) -> list[Domain]:
        return self.hidden_keys

    def __repr__(self):
        return f"{self.name} <{self.edge}, {self.hidden_keys}>"

    def __str__(self):
        return self.__repr__()

    def invert(self) -> Traverse:
        """
        Inverts the traversal
        Returns:
            Traverse: The inverted traversal
        """
        from schema.edge import SchemaEdge

        edge = self.edge
        rev = SchemaEdge.invert(edge)
        return Traverse(rev)


class Expand(RepresentationStep):
    """
    A representation expression that represents an expansion from one node to another
    That is, an edge from X ---> X x Y
    """

    from schema.node import SchemaNode

    def __init__(
        self,
        start_node: SchemaNode,
        end_node: SchemaNode,
        indices: list[int],
        hidden_keys: list[Domain],
    ):
        """
        Initialise a new Expand representation step

        Args:
            start_node (SchemaNode): The node to expand from
            end_node (SchemaNode): The node to expand to
            indices (list[int]): The indices of the start_node in the end_node
            hidden_keys (list[Domain]): The hidden keys required to expand
        """
        super().__init__("EXP")
        self.start_node = start_node
        self.end_node = end_node
        self.indices = indices
        self.hidden_keys = hidden_keys

    def __repr__(self):
        return f"{self.name} <{self.start_node}, {self.end_node}, {self.hidden_keys}>"

    def __str__(self):
        return self.__repr__()

    def invert(self) -> Project:
        """
        Inverts the expansion
        """
        return Project(self.end_node, self.start_node, self.indices)

    def get_hidden_keys(self) -> list[Domain]:
        return self.hidden_keys


class Equate(RepresentationStep):
    """
    A representation expression that represents the traversal of a SchemaEquality edge
    """

    def __init__(self, start_node, end_node):
        """
        Initialise a new Equate representation step

        Args:
            start_node (SchemaNode): The node to equate from
            end_node (SchemaNode): The node to equate to
        """
        super().__init__("EQU")
        self.start_node = start_node
        self.end_node = end_node

    def __repr__(self):
        return f"{self.name} <{self.start_node}, {self.end_node}>"

    def __str__(self):
        return self.__repr__()


class Get(RepresentationStep):
    """
    A representation expression that represents the retrieval of a set of columns,
    or equivalently, the identification of a node in the schema graph
    """

    def __init__(self, columns):
        """
        Initialise a new Get representation step

        Args:
            columns (list[Domain]): The columns to retrieve
        """
        super().__init__("GET")
        self.columns = columns

    def __repr__(self):
        return f"{self.name} <{self.columns}>"

    def __str__(self):
        return self.__repr__()


class Push(RepresentationStep):
    """
    A representation expression that represents duplicating the table at the top of the stack
    """

    def __init__(self):
        super().__init__("PSH")

    def __repr__(self):
        return self.name

    def __str__(self):
        return self.__repr__()


class Merge(RepresentationStep):
    """
    A representation expression that represents merging the two tables at the top of the stack
    on their common columns
    """

    def __init__(self):
        super().__init__("MER")

    def __repr__(self):
        return self.name

    def __str__(self):
        return self.__repr__()


class Pop(RepresentationStep):
    """
    A representation expression that pops the top of the stack
    """

    def __init__(self):
        super().__init__("POP")

    def __repr__(self):
        return self.name

    def __str__(self):
        return self.__repr__()


class Call(RepresentationStep):
    """
    Pushes a new stack frame onto the stack
    """

    def __init__(self):
        super().__init__("CAL")

    def __repr__(self):
        return self.name

    def __str__(self):
        return self.__repr__()


class Return(RepresentationStep):
    """
    Pops the stack frame at the top of the stack
    """

    def __init__(self):
        super().__init__("RET")

    def __repr__(self):
        return self.name

    def __str__(self):
        return self.__repr__()


class Reset(RepresentationStep):
    """
    Resets to the argument the current stack frame was called with
    """

    def __init__(self):
        super().__init__("RST")

    def __repr__(self):
        return self.name

    def __str__(self):
        return self.__repr__()


class Rename(RepresentationStep):
    """
    Renames columns in a table
    """

    def __init__(self, mapping: dict[str, str]):
        super().__init__("RNM")
        self.mapping = mapping

    def __repr__(self):
        return f"{self.name} <{self.mapping}>"

    def __str__(self):
        return self.__repr__()


class Project(RepresentationStep):
    """
    Projects a table onto a subset of its columns
    """

    from schema.node import SchemaNode

    def __init__(
        self, start_node: SchemaNode, end_node: SchemaNode, indices: list[int]
    ):
        """
        Initialise a new Project representation step

        Args:
            start_node (SchemaNode): The node to project from
            end_node (SchemaNode): The node to project to
            indices (list[int]): The indices of the start_node in the end_node
        """
        super().__init__("PRJ")
        self.start_node = start_node
        self.end_node = end_node
        self.indices = indices

    def __repr__(self):
        return f"{self.name} <{self.start_node}, {self.end_node}, {self.indices}>"

    def __str__(self):
        return self.__repr__()

    def invert(self) -> Expand:
        """
        Inverts the projection

        Return:
            Expand: The inverted projection
        """
        from schema.node import SchemaNode

        nodes = SchemaNode.get_constituents(self.start_node)
        hidden_keys = [n for (i, n) in enumerate(nodes) if i not in set(self.indices)]
        return Expand(self.end_node, self.start_node, self.indices, hidden_keys)


class Drop(RepresentationStep):
    """
    Drops a subset of columns from a table
    """

    def __init__(self, columns: list[Domain]):
        super().__init__("DRP")
        self.columns = columns

    def __repr__(self):
        return f"{self.name} <{self.columns}>"

    def __str__(self):
        return self.__repr__()


class Filter(RepresentationStep):
    """
    Filters a table based on a predicate
    """

    def __init__(self, column):
        super().__init__("FLT")
        self.column = column

    def __repr__(self):
        return f"{self.name} <{self.column}>"

    def __str__(self):
        return self.__repr__()


class Sort(RepresentationStep):
    """
    Sorts a table based on a list of columns
    """

    def __init__(self, columns: list[str]):
        super().__init__("SRT")
        self.columns = columns

    def __repr__(self):
        return f"{self.name} <{self.columns}>"

    def __str__(self):
        return self.__repr__()


class End(RepresentationStep):
    """
    A representation expression that represents the end of the intermediate representation
    """

    def __init__(self, left, hidden, right):
        """
        Initialise a new End representation step

        left (list[Domain]): The columns to the left of the marker
        hidden (list[Domain]): The hidden columns
        right (list[Domain]): The columns to the right of the marker
        """
        super().__init__("END")
        self.left = left
        self.hidden = hidden
        self.right = right

    def __repr__(self):
        return f"{self.name} <{self.left}, {self.hidden}, {self.right}>"

    def __str__(self):
        return self.__repr__()
