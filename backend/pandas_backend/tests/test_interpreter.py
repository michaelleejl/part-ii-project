import expecttest
import pandas as pd

from backend.pandas_backend.interpreter import StackPointer
from representation.mapping import Mapping
from schema.edge import SchemaEdge
from schema.equality import SchemaEquality
from schema.node import AtomicNode, SchemaClass
from schema.schema import Schema, SchemaNode
from representation.representation import (
    StartTraversal,
    Traverse,
    Expand,
    Project,
    Equate,
    EndTraversal,
)
from representation.domain import Domain
from frontend.tables.column_type import ColumnType


class TestInterpreter(expecttest.TestCase):

    def initialise(self):
        characters_df = pd.DataFrame(
            {
                "name": ["Luke Skywalker", "Chewbacca", "Han Solo"],
                "homeworld": ["Tatooine", "Kashyyk", "Corellia"],
            }
        ).set_index("name")
        sectors_df = pd.DataFrame(
            {
                "world": ["Coruscant", "Corellia", "Kashyyk", "Naboo", "Tatooine"],
                "sector": ["Core", "Core", "Mid", "Mid", "Outer"],
            }
        ).set_index("world")

        trilogies_df = pd.DataFrame(
            {
                "trilogy": [
                    "Prequel",
                    "Prequel",
                    "Prequel",
                    "Original",
                    "Original",
                    "Original",
                    "Sequel",
                    "Sequel",
                    "Sequel",
                ],
                "episode": [1, 2, 3, 4, 5, 6, 7, 8, 9],
                "director": [
                    "G. Lucas",
                    "G. Lucas",
                    "G. Lucas",
                    "G. Lucas",
                    "I. Kershner",
                    "R. Marquand",
                    "J. J. Abrams",
                    "R. Johnson",
                    "J. J. Abrams",
                ],
            }
        ).set_index(["trilogy", "episode"])

        s = Schema()
        characters = s.insert_dataframe(characters_df)
        sectors = s.insert_dataframe(sectors_df)
        trilogies = s.insert_dataframe(trilogies_df)
        World = SchemaClass("world")
        s.blend(characters["homeworld"], sectors["world"], World)
        return s, characters, sectors, trilogies

    def test_interpreter_startTraversal(self):
        from backend.pandas_backend.interpreter import stt

        s, characters, sectors, trilogies = self.initialise()

        homeworld_column = Domain("homeworld", characters["homeworld"])
        start_columns = [homeworld_column]
        df = pd.DataFrame(
            {
                "character": ["Luke Skywalker", "Chewbacca", "Han Solo"],
                "homeworld": ["Tatooine", "Kashyyk", "Corellia"],
            }
        )
        step = StartTraversal(start_columns)
        sp = StackPointer(0)
        stack, sp = stt(step, s.backend, [df], sp)
        self.assertExpectedInline(
            str(stack[-1]),
            """\
  homeworld         0
0  Tatooine  Tatooine
1   Kashyyk   Kashyyk
2  Corellia  Corellia""",
        )
        self.assertExpectedInline(str(sp), """SP<0>""")

    def test_traversal_ofEdge_withNoHiddenKeys(self):
        from backend.pandas_backend.interpreter import trv

        s, characters, sectors, trilogies = self.initialise()
        start_node = sectors["world"]
        end_node = sectors["sector"]
        edge = Mapping(SchemaEquality(start_node, end_node))
        step = Traverse(edge)
        df = pd.DataFrame(
            {
                "homeworld": ["Tatooine", "Kashyyk", "Corellia"],
                0: ["Tatooine", "Kashyyk", "Corellia"],
            }
        )
        sp = StackPointer(0)
        stack, sp = trv(step, s.backend, [df], sp)
        self.assertExpectedInline(
            str(stack[0]),
            """\
  homeworld      0
0       NaN   Core
1  Corellia   Core
2   Kashyyk    Mid
3       NaN    Mid
4  Tatooine  Outer""",
        )
        self.assertExpectedInline(str(sp), """SP<0>""")

    def test_traversal_ofEdge_withHiddenKeys(self):
        from backend.pandas_backend.interpreter import trv

        s, characters, sectors, trilogies = self.initialise()
        start_node = sectors["sector"]
        end_node = sectors["world"]
        step = Traverse(
            Mapping(
                SchemaEdge(start_node, end_node),
                hidden_keys=[Domain("hiddenWorld", sectors["world"])],
            ),
        )
        df = pd.DataFrame(
            {"sectors": ["Core", "Mid", "Outer"], 0: ["Core", "Mid", "Outer"]}
        )
        stack, sp = trv(step, s.backend, [df], None)
        self.assertExpectedInline(
            str(stack[-1]),
            """\
  sectors          0 hiddenWorld
0    Core  Coruscant   Coruscant
1    Core   Corellia    Corellia
2     Mid    Kashyyk     Kashyyk
3     Mid      Naboo       Naboo
4   Outer   Tatooine    Tatooine""",
        )

    def test_equ_does_nothing(self):
        from backend.pandas_backend.interpreter import equ

        s, characters, sectors, trilogies = self.initialise()
        df = pd.DataFrame({0: ["Tatooine", "Kashyyk", "Corellia"]})
        start_node = AtomicNode("homeworld")
        end_node = AtomicNode("world")
        step = Equate(start_node, end_node)
        stack, sp = equ(step, s.backend, [df], None)
        self.assertExpectedInline(
            str(stack[-1]),
            """\
          0
0  Tatooine
1   Kashyyk
2  Corellia""",
        )

    def test_ent(self):
        from backend.pandas_backend.interpreter import ent

        s, characters, sectors, trilogies = self.initialise()
        step = EndTraversal(
            [Domain("world", sectors["world"])],
        )
        df = pd.DataFrame(
            {
                "sectors": ["Core", "Core", "Mid", "Mid", "Outer"],
                0: ["Coruscant", "Corellia", "Kashyyk", "Naboo", "Tatooine"],
                "hiddenWorld": [
                    "Coruscant",
                    "Corellia",
                    "Kashyyk",
                    "Naboo",
                    "Tatooine",
                ],
            }
        )
        bs = pd.DataFrame({"sectors": ["Core", "Mid", "Outer"]})
        stack, sp = ent(step, s.backend, [bs, df], None)
        self.assertExpectedInline(
            str(stack[-1]),
            """\
  sectors      world hiddenWorld
0    Core  Coruscant   Coruscant
1    Core   Corellia    Corellia
2     Mid    Kashyyk     Kashyyk
3     Mid      Naboo       Naboo
4   Outer   Tatooine    Tatooine""",
        )

    def test_mer(self):
        from backend.pandas_backend.interpreter import mer

        s, characters, sectors, trilogies = self.initialise()
        step = EndTraversal(
            [Domain("world", sectors["world"])],
        )
        df = pd.DataFrame(
            {
                "world": ["Coruscant", "Corellia", "Kashyyk", "Naboo", "Tatooine"],
                "sector": ["Core", "Core", "Mid", "Mid", "Outer"],
            }
        )
        bs = pd.DataFrame(
            {
                "world": ["Tatooine", "Kashyyk", "Corellia"],
                "character": ["Luke Skywalker", "Chewbacca", "Han Solo"],
            }
        )
        stack, sp = mer(step, s.backend, [bs, df], None)
        self.assertExpectedInline(
            str(stack[-1]),
            """\
       world sector       character
0  Coruscant   Core             NaN
1   Corellia   Core        Han Solo
2    Kashyyk    Mid       Chewbacca
3      Naboo    Mid             NaN
4   Tatooine  Outer  Luke Skywalker""",
        )
