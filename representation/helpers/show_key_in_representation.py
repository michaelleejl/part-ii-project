from frontend.domain import Domain
from representation.representation import RepresentationStep, StartTraversal, Project, Expand, EndTraversal


def show_key_in_representation(representation: list[RepresentationStep], key: Domain) -> list[RepresentationStep]:
    for step in representation:
        if isinstance(step, StartTraversal):
            print(step.start_columns)
        elif isinstance(step, Expand):
            print(step.hidden_keys)
        elif isinstance(step, Traverse):

        elif isinstance(step, EndTraversal):
            print(step.end_columns)
        else:
            print(step)