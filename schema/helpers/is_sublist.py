from schema.helpers.find_index import find_index


def is_sublist(list1, list2):
    if len(list1) > len(list2):
        return False
    start_index = 0
    for item in list1:
        new_index = find_index(item, list2[start_index:]) + start_index
        if start_index > new_index:
            return False
        start_index = new_index
    return True