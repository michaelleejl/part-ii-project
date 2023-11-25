class UpdatingDataShouldPreserveColumnsException(Exception):
    def __init__(self, additional_columns, missing_columns):
        msg = f"Columns do not match. Data to update has additional columns: " \
              f"{additional_columns} and missing columns: {missing_columns}"
        super().__init__(msg)


class KeyDuplicationException(Exception):
    def __init__(self, msg):
        super().__init__(f"Duplicate key/s detected: {msg}.")