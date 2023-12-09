import expecttest
import numpy as np
import pandas as pd

from schema import Schema, SchemaNode


class TestEx4(expecttest.TestCase):
    def test_ex4(self):
        s = Schema()

        person = pd.read_csv("./csv/roles/person.csv").set_index("person")
        task = pd.read_csv("./csv/roles/task.csv").set_index("task")

        s.insert_dataframe(person, "person")
        s.insert_dataframe(task, "task")

        p_role = SchemaNode("role", cluster="person")
        t_role = SchemaNode("role", cluster="task")

        s.blend(p_role, t_role, under ="Role")

        # ========================================================================
        # ========================================================================

        # SCHEMA:
        # person ---> role <--- task

        # GOAL: I want to know, for each person, what tasks they may perform

        t1 = s.get(["person.person"])
        print(t1)

        # [person.person || ]
        #  Steve
        #  Tom
        #  Harry
        #  Dick

        t2 = t1.infer(["person.person"], "task.task")
        print(t2)
        # [person.person || task.task]
        #  Steve         || [funding, investment, budget]
        #  Tom           || [research]
        #  Dick          || [manpower]

        # Expose the hidden key
        t3 = t2.show("task.task")
        print(t3)
        # [person.person task.task || task.task]
        #  Steve         funding   || funding
        #  Steve         budget    || budget
        #  Tom           research  || research
        #  Dick          manpower  || manpower

