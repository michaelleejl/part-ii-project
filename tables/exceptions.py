class KeyMismatchException(Exception):
    def __init__(self, keys1, keys2):
        super().__init__(f"Key mismatch exception: no way to reconcile hidden keys {keys1} and {keys2}")


class ColumnsNeedToBeUniqueException(Exception):
    def __init__(self):
        super().__init__("Columns need to be unique")


class ColumnsNeedToBeInTableException(Exception):
    def __init__(self):
        super().__init__("All columns need to be in table")


class ColumnsCannotAlreadyBeInTableException(Exception):
    def __init__(self):
        super().__init__("Clumns cannot already be in table")


class ColumnsNeedToBeKeysException(Exception):
    def __init__(self):
        super().__init__("All columns need to be keys")


class ColumnsNeedToBeValuesException(Exception):
    def __init__(self):
        super().__init__("All columns need to be keys")


class ColumnsNeedToBeHiddenException(Exception):
    def __init__(self):
        super().__init__("All columns need to be keys")


class ColumnsNeedToBeInTableAndVisibleException(Exception):
    def __init__(self):
        super().__init__("All columns need to be in the table and visible")

class IntermediateRepresentationMustHaveEndMarkerException(Exception):
    def __init__(self):
        super().__init__("Intermediate representation must end with End marker")