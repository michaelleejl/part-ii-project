def get_indices_of_sublist(sublist, list):
    idxs = []
    assert len(sublist) <= len(list)
    i = 0
    j = 0
    while i < len(sublist):
        if sublist[i] == list[j]:
            idxs += [i]
            i += 1
        j += 1
    return idxs
