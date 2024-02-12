from schema.helpers.find_index import find_index
from representation.representation import *


def perform_renaming(old_name: str, new_name: str):
    def renaming(domains: list[Domain]):
        idx = find_index(old_name, [d.name for d in domains])
        if idx < 0:
            return domains
        else:
            return (
                domains[:idx]
                + [Domain(new_name, domains[idx].node)]
                + domains[idx + 1 :]
            )

    return renaming


def rename_column_in_step(step: RepresentationStep, old_name: str, new_name: str):
    renaming_function = perform_renaming(old_name, new_name)
    match step.name:
        case "GET":
            assert isinstance(step, Get)
            return Get(renaming_function(step.columns))
        case "STT":
            assert isinstance(step, StartTraversal)
            return StartTraversal(renaming_function(step.start_columns))
        case "ENT":
            assert isinstance(step, EndTraversal)
            return EndTraversal(renaming_function(step.end_columns))
        case "TRV":
            assert isinstance(step, Traverse)
            return Traverse(step.edge, renaming_function(step.columns))
        case "EXP":
            assert isinstance(step, Expand)
            return Expand(
                step.start_node,
                step.end_node,
                step.indices,
                renaming_function(step.hidden_keys),
            )
        case "PRJ":
            assert isinstance(step, Project)
            return Project(
                step.start_node,
                step.end_node,
                step.indices,
            )
        case "DRP":
            assert isinstance(step, Drop)
            return Drop(renaming_function(step.columns))
        case _:
            return step


def rename_column_in_representation(
    representation: list[RepresentationStep], old_name: str, new_name: str
):
    steps = [rename_column_in_step(step, old_name, new_name) for step in representation]
    return steps
