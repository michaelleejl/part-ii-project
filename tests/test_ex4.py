import expecttest
import numpy as np
import pandas as pd

from schema import Schema, SchemaNode


class TestEx4(expecttest.TestCase):

    def initialise(self) -> Schema:
        s = Schema()

        person = pd.read_csv("./csv/roles/person.csv").set_index("person")
        task = pd.read_csv("./csv/roles/task.csv").set_index("task")

        s.insert_dataframe(person, "person")
        s.insert_dataframe(task, "task")

        p_role = SchemaNode("role", cluster="person")
        t_role = SchemaNode("role", cluster="task")

        s.blend(p_role, t_role, under="Role")
        return s

        # SCHEMA:
        # person ---> role <--- task

# GOAL: I want to know, for each person, what tasks they may perform

    def test_ex4_goal1_step1_get(self):
        s = self.initialise()
        t1 = s.get(["person.person"])
        self.assertExpectedInline(str(t1), """\
[person.person || ]
Empty DataFrame
Columns: []
Index: [Steve, Tom, Harry, Dick]

""")

        # [person.person || ]
        #  Steve
        #  Tom
        #  Harry
        #  Dick

    def test_ex4_goal1_step2_infer(self):
        s = self.initialise()
        t1 = s.get(["person.person"])
        t2 = t1.infer(["person.person"], "task.task")
        self.assertExpectedInline(str(t2), """\
[person.person || task.task]
                                   task.task
person.person                               
Dick                              [manpower]
Tom                               [research]
Steve          [funding, investment, budget]
2 keys hidden

""")
        # [person.person || task.task]
        #  Steve         || [funding, investment, budget]
        #  Tom           || [research]
        #  Dick          || [manpower]

    def test_ex4_goal1_step3_show(self):
        s = self.initialise()
        t1 = s.get(["person.person"])
        t2 = t1.infer(["person.person"], "task.task")
        t3 = t2.show("task.task")
        self.assertExpectedInline(str(t3), """\
[person.person task.task || task.task]
                           task.task
person.person task.task             
Dick          manpower      manpower
Tom           research      research
Steve         funding        funding
              investment  investment
              budget          budget
1 keys hidden
2 values hidden

""")
        # [person.person task.task || task.task]
        #  Steve         funding   || funding
        #  Steve         budget    || budget
        #  Tom           research  || research
        #  Dick          manpower  || manpower

    def test_ex4_goal1_step4_hideAfterShow(self):
        s = self.initialise()
        t1 = s.get(["person.person"])
        t2 = t1.infer(["person.person"], "task.task")
        print(t2)
        t3 = t2.show("task.task")
        print(t3)
        t4 = t3.hide("task.task")
        print(t4)
        self.assertExpectedInline(str(t4), """\
[person.person || task.task]
                                   task.task
person.person                               
Dick                              [manpower]
Tom                               [research]
Steve          [funding, investment, budget]
2 keys hidden

""")

    def test_ex4_goal1_step4_showAfterHideAfterShow(self):
        s = self.initialise()
        t1 = s.get(["person.person"])
        t2 = t1.infer(["person.person"], "task.task")
        print(t2)
        t3 = t2.show("task.task")
        print(t3)
        t4 = t3.hide("task.task")
        print(t4)
        t5 = t4.show("task.task")
        self.maxDiff = None
        self.assertExpectedInline(str(t5), """\
[person.person task.task || task.task]
                           task.task
person.person task.task             
Dick          manpower      manpower
Tom           research      research
Steve         funding        funding
              investment  investment
              budget          budget
1 keys hidden
2 values hidden

""")

