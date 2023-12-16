import expecttest
import pandas as pd

from schema import Schema, SchemaNode
from tables.derivation import StartTraversal, Traverse, Expand, Project, Equate
from tables.raw_column import RawColumn, ColumnType
from tables.table import Table


class TestInterpreter(expecttest.TestCase):

    def initialise(self):
        characters_df = pd.DataFrame({"name": ["Luke Skywalker", "Chewbacca", "Han Solo"],
                                      "homeworld": ["Tatooine", "Kashyyk", "Corellia"]}).set_index("name")
        sectors_df = pd.DataFrame({"world": ["Coruscant", "Corellia", "Kashyyk", "Naboo", "Tatooine"],
                                  "sector": ["Core", "Core", "Mid", "Mid", "Outer"]}).set_index("world")

        trilogies_df = pd.DataFrame({"trilogy": ["Prequel", "Prequel", "Prequel",
                                                 "Original", "Original", "Original",
                                                 "Sequel", "Sequel", "Sequel"],
                                     "episode": [1, 2, 3, 4, 5, 6, 7, 8, 9],
                                     "director": ["G. Lucas", "G. Lucas", "G. Lucas",
                                                  "G. Lucas", "I. Kershner", "R. Marquand",
                                                  "J. J. Abrams", "R. Johnson", "J. J. Abrams"]
                                     }).set_index(["trilogy", "episode"])

        s = Schema()
        s.insert_dataframe(characters_df, "characters")
        s.insert_dataframe(sectors_df, "sectors")
        s.insert_dataframe(trilogies_df, "trilogies")
        s.blend(SchemaNode("homeworld", cluster="characters"), SchemaNode("world", cluster="sectors"))
        return s

    def test_interpreter_startTraversal(self):
        from backend.pandas_backend.interpreter import stt
        s = self.initialise()
        character_column = RawColumn("character", SchemaNode("name", cluster="characters"), [], ColumnType.KEY)
        homeworld_column = RawColumn("homeworld", SchemaNode("homeworld", cluster="characters"), [character_column], ColumnType.VALUE)
        start_columns = [homeworld_column]
        explicit_keys = [character_column, homeworld_column]
        df = pd.DataFrame({"character": ["Luke Skywalker", "Chewbacca", "Han Solo"],
                           "homeworld": ["Tatooine", "Kashyyk", "Corellia"]})
        step = StartTraversal(start_columns, explicit_keys)
        result = stt(step, s.backend, df, lambda x: x, [], [])
        self.assertExpectedInline(str(result[0]), """\
  homeworld         0
0  Tatooine  Tatooine
1   Kashyyk   Kashyyk
2  Corellia  Corellia""")
        self.assertExpectedInline(str(result[2]), """\
[        character homeworld
0  Luke Skywalker  Tatooine
1       Chewbacca   Kashyyk
2        Han Solo  Corellia]""")
        self.assertExpectedInline(str(result[3]), """['character', 'homeworld']""")

    def test_traversal_ofEdge_withNoHiddenKeys(self):
        from backend.pandas_backend.interpreter import trv
        s = self.initialise()
        start_node = SchemaNode("world", cluster="sectors")
        end_node = SchemaNode("sector", cluster="sectors")
        step = Traverse(start_node, end_node)
        character_column = RawColumn("character", SchemaNode("name", cluster="characters"), [], ColumnType.KEY)
        homeworld_column = RawColumn("homeworld", SchemaNode("homeworld", cluster="characters"), [character_column],
                                     ColumnType.VALUE)
        df = pd.DataFrame({"homeworld": ["Tatooine", "Kashyyk", "Corellia"], 0: ["Tatooine", "Kashyyk", "Corellia"]})
        result = trv(step, s.backend, df, lambda x: x, [df], [character_column.name, homeworld_column.name])
        self.assertExpectedInline(str(result[0]), """\
  homeworld      0
0       NaN   Core
1  Corellia   Core
2   Kashyyk    Mid
3       NaN    Mid
4  Tatooine  Outer""")
        self.assertExpectedInline(str(result[2]), """\
[  homeworld         0
0  Tatooine  Tatooine
1   Kashyyk   Kashyyk
2  Corellia  Corellia]""")
        self.assertExpectedInline(str(result[3]), """['character', 'homeworld']""")

    def test_expansion_ofDomain_withNoHiddenKey(self):
        from backend.pandas_backend.interpreter import trv
        s = self.initialise()
        start_node = SchemaNode("sector", cluster="sectors")
        end_node = SchemaNode("world", cluster="sectors")
        step = Traverse(start_node, end_node, [end_node], [RawColumn("world_hidden", end_node, [], ColumnType.KEY)])
        sectors_column = RawColumn("sectors", SchemaNode("sector", cluster="sectors"), [], ColumnType.KEY)
        df = pd.DataFrame(
            {"sectors": ["Core", "Mid", "Outer"], 0: ["Core", "Mid", "Outer"]})
        result = trv(step, s.backend, df, lambda x: x, [df], [sectors_column])
        self.assertExpectedInline(str(result[0]), """\
  sectors          0 world_hidden
0    Core  Coruscant    Coruscant
1    Core   Corellia     Corellia
2     Mid    Kashyyk      Kashyyk
3     Mid      Naboo        Naboo
4   Outer   Tatooine     Tatooine""")
        self.assertExpectedInline(str(result[2]), """\
[  sectors      0
0    Core   Core
1     Mid    Mid
2   Outer  Outer]""")
        self.assertExpectedInline(str(result[3]), """[sectors, 'world_hidden']""")

    def test_proj_ofEdge_withNoHiddenKey(self):
        from backend.pandas_backend.interpreter import prj
        s = self.initialise()
        trilogy_column = RawColumn("trilogy", SchemaNode("trilogy", cluster="trilogies"), [], ColumnType.KEY)
        episode_column = RawColumn("trilogy", SchemaNode("episode", cluster="trilogies"), [], ColumnType.KEY)
        df = pd.DataFrame({"trilogy": ["Prequel", "Prequel", "Prequel",
                                       "Original", "Original", "Original",
                                       "Sequel", "Sequel", "Sequel"],
                           "episode": [1, 2, 3, 4, 5, 6, 7, 8, 9],
                           0: ["Prequel", "Prequel", "Prequel",
                               "Original", "Original", "Original",
                               "Sequel", "Sequel", "Sequel"],
                           1: [1, 2, 3, 4, 5, 6, 7, 8, 9],
                           })
        start_node = SchemaNode.product([SchemaNode("trilogy", cluster="trilogies"), SchemaNode("episode", cluster="trilogies")])
        end_node = SchemaNode("episode", cluster="trilogies")
        step = Project(start_node, end_node)
        result = prj(step, s.backend, df, lambda x: x, [df], [trilogy_column])
        self.assertExpectedInline(str(result[0]), """\
    trilogy  episode  0
0   Prequel        1  1
1   Prequel        2  2
2   Prequel        3  3
3  Original        4  4
4  Original        5  5
5  Original        6  6
6    Sequel        7  7
7    Sequel        8  8
8    Sequel        9  9""")
        self.assertExpectedInline(str(result[2]), """\
[    trilogy  episode         0  1
0   Prequel        1   Prequel  1
1   Prequel        2   Prequel  2
2   Prequel        3   Prequel  3
3  Original        4  Original  4
4  Original        5  Original  5
5  Original        6  Original  6
6    Sequel        7    Sequel  7
7    Sequel        8    Sequel  8
8    Sequel        9    Sequel  9]""")
        self.assertExpectedInline(str(result[3]), """[trilogy]""")

    def test_proj_ofEdge_withHiddenKey(self):
        from backend.pandas_backend.interpreter import prj
        s = self.initialise()
        trilogy_column = RawColumn("trilogy", SchemaNode("trilogy", cluster="trilogies"), [], ColumnType.KEY)
        episode_column = RawColumn("trilogy", SchemaNode("episode", cluster="trilogies"), [], ColumnType.KEY)
        df = pd.DataFrame({
                           0: ["Prequel", "Prequel", "Prequel",
                               "Original", "Original", "Original",
                               "Sequel", "Sequel", "Sequel"],
                           1: [1, 2, 3, 4, 5, 6, 7, 8, 9],
                           })
        start_node = SchemaNode.product(
            [SchemaNode("trilogy", cluster="trilogies"), SchemaNode("episode", cluster="trilogies")])
        end_node = SchemaNode("episode", cluster="trilogies")
        step = Project(start_node, end_node, [SchemaNode("trilogy", cluster="trilogies")], [trilogy_column])
        result = prj(step, s.backend, df, lambda x: x, [df], [])
        self.assertExpectedInline(str(result[0]), """\
   0   trilogy
0  1   Prequel
1  2   Prequel
2  3   Prequel
3  4  Original
4  5  Original
5  6  Original
6  7    Sequel
7  8    Sequel
8  9    Sequel""")
        self.assertExpectedInline(str(result[2]), """\
[          0  1
0   Prequel  1
1   Prequel  2
2   Prequel  3
3  Original  4
4  Original  5
5  Original  6
6    Sequel  7
7    Sequel  8
8    Sequel  9]""")
        self.assertExpectedInline(str(result[3]), """['trilogy']""")

    def test_exp_ofEdge_withNoHiddenKey(self):
        from backend.pandas_backend.interpreter import exp
        s = self.initialise()
        trilogy_column = RawColumn("trilogy", SchemaNode("trilogy", cluster="trilogies"), [], ColumnType.KEY)
        df = pd.DataFrame({"trilogy": ["Prequel", "Original", "Sequel"], 0: ["Prequel", "Original", "Sequel"]})
        start_node = SchemaNode("trilogy", cluster="trilogies")
        end_node = SchemaNode.product([start_node, SchemaNode("episode", cluster="trilogies")])
        step = Expand(start_node, end_node)
        result = exp(step, s.backend, df, lambda x: x, [df], [trilogy_column])
        self.assertExpectedInline(str(result[0]), """\
     trilogy         0  1
0    Prequel   Prequel  1
1    Prequel   Prequel  2
2    Prequel   Prequel  3
3    Prequel   Prequel  4
4    Prequel   Prequel  5
5    Prequel   Prequel  6
6    Prequel   Prequel  7
7    Prequel   Prequel  8
8    Prequel   Prequel  9
9   Original  Original  1
10  Original  Original  2
11  Original  Original  3
12  Original  Original  4
13  Original  Original  5
14  Original  Original  6
15  Original  Original  7
16  Original  Original  8
17  Original  Original  9
18    Sequel    Sequel  1
19    Sequel    Sequel  2
20    Sequel    Sequel  3
21    Sequel    Sequel  4
22    Sequel    Sequel  5
23    Sequel    Sequel  6
24    Sequel    Sequel  7
25    Sequel    Sequel  8
26    Sequel    Sequel  9""")
        self.assertExpectedInline(str(result[2]), """\
[    trilogy         0
0   Prequel   Prequel
1  Original  Original
2    Sequel    Sequel]""")
        self.assertExpectedInline(str(result[3]), """[trilogy]""")

    def test_exp_ofEdge_withHiddenKey(self):
        from backend.pandas_backend.interpreter import exp
        s = self.initialise()
        trilogy_column = RawColumn("trilogy", SchemaNode("trilogy", cluster="trilogies"), [], ColumnType.KEY)
        episode_column = RawColumn("episode", SchemaNode("episode", cluster="trilogies"), [], ColumnType.KEY)
        df = pd.DataFrame({"trilogy": ["Prequel", "Original", "Sequel"], 0: ["Prequel", "Original", "Sequel"]})
        start_node = SchemaNode("trilogy", cluster="trilogies")
        end_node = SchemaNode.product([start_node, SchemaNode("episode", cluster="trilogies")])
        step = Expand(start_node, end_node, [SchemaNode("episode", cluster="trilogies")], [episode_column])
        result = exp(step, s.backend, df, lambda x: x, [df], [trilogy_column])
        self.assertExpectedInline(str(result[0]), """\
     trilogy         0  1  episode
0    Prequel   Prequel  1        1
1    Prequel   Prequel  2        2
2    Prequel   Prequel  3        3
3    Prequel   Prequel  4        4
4    Prequel   Prequel  5        5
5    Prequel   Prequel  6        6
6    Prequel   Prequel  7        7
7    Prequel   Prequel  8        8
8    Prequel   Prequel  9        9
9   Original  Original  1        1
10  Original  Original  2        2
11  Original  Original  3        3
12  Original  Original  4        4
13  Original  Original  5        5
14  Original  Original  6        6
15  Original  Original  7        7
16  Original  Original  8        8
17  Original  Original  9        9
18    Sequel    Sequel  1        1
19    Sequel    Sequel  2        2
20    Sequel    Sequel  3        3
21    Sequel    Sequel  4        4
22    Sequel    Sequel  5        5
23    Sequel    Sequel  6        6
24    Sequel    Sequel  7        7
25    Sequel    Sequel  8        8
26    Sequel    Sequel  9        9""")
        self.assertExpectedInline(str(result[2]), """\
[    trilogy         0
0   Prequel   Prequel
1  Original  Original
2    Sequel    Sequel]""")
        self.assertExpectedInline(str(result[3]), """[trilogy, 'episode']""")

    def test_equ_does_nothing(self):
        from backend.pandas_backend.interpreter import equ
        s = self.initialise()
        df = pd.DataFrame({0: ["Tatooine", "Kashyyk", "Corellia"]})
        start_node = SchemaNode("homeworld", cluster="characters")
        end_node = SchemaNode("world", cluster="sectors")
        step = Equate(start_node, end_node)
        result = equ(step, s.backend, df, lambda x: x, [df], [])
        self.assertExpectedInline(str(result[0]), """\
          0
0  Tatooine
1   Kashyyk
2  Corellia""")
        self.assertExpectedInline(str(result[2]), """\
[          0
0  Tatooine
1   Kashyyk
2  Corellia]""")
        self.assertExpectedInline(str(result[3]), """[]""")





