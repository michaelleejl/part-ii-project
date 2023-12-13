import expecttest
import pandas as pd

from schema import Schema, SchemaNode, NoShortestPathBetweenNodesException


class TestEx5(expecttest.TestCase):
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

# GOAL 2
# [person task || role]
# Does role depend on person or task - depends on what you call infer on!

    def test_ex5_step1_get(self):
        s = self.initialise()
        t1 = s.get(["person.person", "task.task"])
        self.assertExpectedInline(str(t1), """\
[person.person task.task || ]
Empty DataFrame
Columns: []
Index: [(Steve, logistics), (Steve, manpower), (Steve, research), (Steve, funding), (Steve, marketing), (Steve, investment), (Steve, budget), (Tom, logistics), (Tom, manpower), (Tom, research), (Tom, funding), (Tom, marketing), (Tom, investment), (Tom, budget), (Harry, logistics), (Harry, manpower), (Harry, research), (Harry, funding), (Harry, marketing), (Harry, investment), (Harry, budget), (Dick, logistics), (Dick, manpower), (Dick, research), (Dick, funding), (Dick, marketing), (Dick, investment), (Dick, budget)]

""")
        # [person.person task.task || ]
        # Steve          logistics
        # Steve          manpower
        # Steve          research
        # ...
        # Dick           investment
        # Dick           budget

    def test_ex5_step2_infer(self):
        s = self.initialise()
        t1 = s.get(["person.person", "task.task"])
        t2 = t1.infer(["task.task"], "Role")
        self.assertExpectedInline(str(t2), """\
[person.person task.task || Role]
                         Role
person.person task.task      
Steve         logistics   COO
Tom           logistics   COO
Harry         logistics   COO
Dick          logistics   COO
Steve         manpower    CEO
Tom           manpower    CEO
Harry         manpower    CEO
Dick          manpower    CEO
Steve         research    CTO
Tom           research    CTO
Harry         research    CTO
Dick          research    CTO
Steve         funding     CFO
Tom           funding     CFO
Harry         funding     CFO
Dick          funding     CFO
Steve         marketing   COO
Tom           marketing   COO
Harry         marketing   COO
Dick          marketing   COO
Steve         investment  CFO
Tom           investment  CFO
Harry         investment  CFO
Dick          investment  CFO
Steve         budget      CFO
Tom           budget      CFO
Harry         budget      CFO
Dick          budget      CFO

""")
        # [person.person task.task   ||  Role]
        # Steve          logistics   ||  COO
        # Steve          manpower    ||  CEO
        # Steve          research    ||  CTO
        # ...
        # Dick           investment  ||  CFO
        # Dick           budget      ||  CFO

        # This is a lookup table.
        # If I know Steve is in charge of manpower, then he must be the CEO

    def test_ex5_step3_infer(self):
        # Or perhaps I can ask the other question - if I know person does task,
        # and I know their role, what can I infer about the role demanded by the task?
        s = self.initialise()
        t1 = s.get(["person.person", "task.task"])
        t3 = t1.infer(["person.person"], "Role")
        self.assertExpectedInline(str(t3), """\
[person.person task.task || Role]
                         Role
person.person task.task      
Steve         logistics   CFO
              manpower    CFO
              research    CFO
              funding     CFO
              marketing   CFO
              investment  CFO
              budget      CFO
Tom           logistics   CTO
              manpower    CTO
              research    CTO
              funding     CTO
              marketing   CTO
              investment  CTO
              budget      CTO
Harry         logistics   CAO
              manpower    CAO
              research    CAO
              funding     CAO
              marketing   CAO
              investment  CAO
              budget      CAO
Dick          logistics   CEO
              manpower    CEO
              research    CEO
              funding     CEO
              marketing   CEO
              investment  CEO
              budget      CEO

""")
        # [person.person task.task   ||  Role]
        # Steve          logistics   ||  CFO
        # Steve          manpower    ||  CFO
        # Steve          research    ||  CFO
        # ...
        # Dick           investment  ||  CEO
        # Dick           budget      ||  CEO

        # Note that in the second case, task is a weak key for role,
        # whereas in the first case, person is a weak key for role.

    def test_ex5_step3_stressTest(self):
        # STRESS TEST
        s = self.initialise()
        t1 = s.get(["person.person", "task.task"])
        self.assertExpectedRaisesInline(NoShortestPathBetweenNodesException, lambda: t1.infer(["person.person", "task.task"], "Role"),"""No paths found between nodes person;task and Role.If the path involves a projection that isn't the last edge in the path,The projection will need to be specified as a waypoint.""")
        # There is no path from person.person x task.task to Role, only Role x Role.
        # This should throw an error