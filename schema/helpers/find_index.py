def find_index(item, in_list: list[any]):
    try:
        return in_list.index(item)
    except ValueError:
        return -1