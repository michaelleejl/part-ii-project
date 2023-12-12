import expecttest
import pandas as pd

from schema import Schema, SchemaNode


class TestEx5(expecttest.TestCase):
    def test_ex5(self):
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

        # GOAL 2
        # [person task || role]
        # Does role depend on person or task - depends on what you call infer on!

        t11 = s.get(["person.person", "task.task"])
        print(t11)
        # [person.person task.task || ]
        # Steve          logistics
        # Steve          manpower
        # Steve          research
        # ...
        # Dick           investment
        # Dick           budget


        t12 = t11.infer(["task.task"], "Role")
        print(t12)
        # [person.person task.task   ||  Role]
        # Steve          logistics   ||  COO
        # Steve          manpower    ||  CEO
        # Steve          research    ||  CTO
        # ...
        # Dick           investment  ||  CFO
        # Dick           budget      ||  CFO

        # This is a lookup table.
        # If I know Steve is in charge of manpower, then he must be the CEO

        # Or perhaps I can ask the other question - if I know person does task,
        # and I know their role, what can I infer about the role demanded by the task?

        t13 = t11.infer(["person.person"], "Role")
        print(t13)
        # [person.person task.task   ||  Role]
        # Steve          logistics   ||  CFO
        # Steve          manpower    ||  CFO
        # Steve          research    ||  CFO
        # ...
        # Dick           investment  ||  CEO
        # Dick           budget      ||  CEO

        # Note that in the second case, task is a weak key for role,
        # whereas in the first case, person is a weak key for role.

        # STRESS TEST
        # What if the user does
        _ = t11.infer(["person.person", "task.task"], "Role")
        # There is no path from person.person x task.task to Role, only Role x Role.
        # This should throw an error