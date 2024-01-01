import expecttest
import numpy as np
import pandas as pd

from schema import Schema, SchemaNode


class TestEx4(expecttest.TestCase):

    def initialise(self):
        s = Schema()

        person_df = pd.read_csv("./csv/roles/person.csv").set_index("person")
        task_df = pd.read_csv("./csv/roles/task.csv").set_index("task")

        person = s.insert_dataframe(person_df)
        task = s.insert_dataframe(task_df)

        p_role = person["role"]
        t_role = task["role"]

        Role = s.create_class("Role")

        s.blend(p_role, t_role, Role)
        return s, person, task, Role

        # SCHEMA:
        # person ---> role <--- task

# GOAL: I want to know, for each person, what tasks they may perform

    def test_ex4_goal1_step1_get(self):
        s, person, task, Role = self.initialise()
        t1 = s.get([person["person"]])
        self.assertExpectedInline(str(t1), """\
[person || ]
Empty DataFrame
Columns: []
Index: []
4 keys hidden

""")

        # [person.person || ]
        #  Steve
        #  Tom
        #  Harry
        #  Dick

    def test_ex4_goal1_step2_infer(self):
        s, person, task, Role = self.initialise()
        t1 = s.get([person["person"]])
        t2 = t1.infer(["person"], task["task"])
        self.assertExpectedInline(str(t2), """\
[person || task]
                                 task
person                               
Dick                       [manpower]
Tom                        [research]
Steve   [funding, investment, budget]
1 keys hidden

""")
        # [person.person || task.task]
        #  Steve         || [funding, investment, budget]
        #  Tom           || [research]
        #  Dick          || [manpower]

    def test_ex4_goal1_step3_show(self):
        s, person, task, Role = self.initialise()
        t1 = s.get([person["person"]])
        t2 = t1.infer(["person"], task["task"])
        t3 = t2.show("task_1")
        self.assertExpectedInline(str(t3), """\
[person task_1 || task]
                         task
person task_1                
Dick   manpower      manpower
Tom    research      research
Steve  funding        funding
       investment  investment
       budget          budget
23 keys hidden

""")
        # [person.person task.task || task.task]
        #  Steve         funding   || funding
        #  Steve         budget    || budget
        #  Tom           research  || research
        #  Dick          manpower  || manpower

    def test_ex4_goal1_step4_hideAfterShow(self):
        s, person, task, Role = self.initialise()
        t1 = s.get([person["person"]])
        t2 = t1.infer(["person"], task["task"])
        t3 = t2.show("task_1")
        t4 = t3.hide("task_1")
        print(t4)
        self.assertExpectedInline(str(t4), """\
[person || task]
                                 task
person                               
Dick                       [manpower]
Tom                        [research]
Steve   [funding, investment, budget]
1 keys hidden

""")

    def test_ex4_goal1_step4_showAfterHideAfterShow(self):
        s, person, task, Role = self.initialise()
        t1 = s.get([person["person"]])
        t2 = t1.infer(["person"], task["task"])
        t3 = t2.show("task_1")
        t4 = t3.hide("task_1")
        t5 = t4.show("task_1")
        self.maxDiff = None
        self.assertExpectedInline(str(t5), """\
[person task_1 || task]
                         task
person task_1                
Dick   manpower      manpower
Tom    research      research
Steve  funding        funding
       investment  investment
       budget          budget
23 keys hidden

""")

