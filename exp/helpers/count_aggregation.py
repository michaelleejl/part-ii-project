def count_aggregation(column: int, aggregated_over: dict[int, int]) -> dict[int, int]:
    if column not in aggregated_over:
        aggregated_over[column] = 0
    aggregated_over[column] += 1
    return aggregated_over
