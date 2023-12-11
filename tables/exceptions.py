class KeyMismatchException(Exception):
    def __init__(self, keys1, keys2):
        super().__init__(f"Key mismatch exception: no way to reconcile hidden keys {keys1} and {keys2}")
