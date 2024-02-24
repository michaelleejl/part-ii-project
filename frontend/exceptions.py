class MustCurryAtSourceException(Exception):
    def __init__(self):
        super().__init__("All keys to be curried must be in the source node")


class MustUncurryHiddenKeyException(Exception):
    def __init__(self):
        super().__init__("All keys to be uncurried must be in the hidden keys")


class MustForwardAtSourceException(Exception):
    def __init__(self):
        super().__init__("All keys to be forwarded must be in the source node")