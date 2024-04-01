from representation.domain import Domain
from schema.helpers.find_index import find_index


def convert_key_to_idx_and_update_parameters(
    key: Domain, parameters: list[Domain]
) -> tuple[int, list[Domain]]:
    idx = find_index(key, parameters)

    if idx == -1:
        idx = len(parameters)
        parameters += [key]

    return idx, parameters
