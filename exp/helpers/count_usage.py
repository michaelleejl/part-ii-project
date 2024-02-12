def count_usages(column: int, usages: dict[int, int]) -> dict[int, int]:
    if column not in usages:
        usages[column] = 0
    usages[column] += 1
    return usages
