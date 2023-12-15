def list_difference(list1, list2):
    i = 0
    j = 0
    diff = []
    while i < len(list1) and j < len(list2):
        a = list1[i]
        b = list2[j]
        if a != b:
            diff += [b]
            j += 1
        else:
            i += 1
            j += 1
    return diff
