import expecttest
import pandas as pd

from schema.schema import Schema


class TestZoo(expecttest.TestCase):

    def initialise(self):
        schema = Schema()
        trips = pd.read_csv("./csv/zoo/trips.csv").set_index("trip_id")
        print(trips)
        trips = schema.insert_dataframe(trips)
        return schema, trips["trip_id"], trips["hr"], trips["destination"]

    def test_zoo_1(self):
        schema, trip_id, hr, destination = self.initialise()
        # Assume you have trip id
        # From tripid you can infer (join) hour and infer (join) destination
        # infer is just a join that preserves derivation information (what it was joined on)
        t1 = (
            schema.get(trip_id=trip_id)
            .infer(["trip_id"], hr)
            .infer(["trip_id"], destination)
        )
        self.assertExpectedInline(
            str(t1),
            """\
[trip_id || hr destination]
         hr destination
trip_id                
1         7         Zoo
2         7         CBD
3         8         Zoo
4         8         Zoo
5         8         Uni
6         9        Cafe
7        10         Uni
8        11      School
9        12  Restaurant
10       13        Home
11       14        Home
12       15        Cafe
13       16        Cafe
14       17         Zoo
15       17         Zoo
16       17        Home
17       17       Shops
18       18       Beach
19       19  Restaurant
20       20         Bar

""",
        )

    def test_zoo_2(self):
        schema, trip_id, hr, destination = self.initialise()
        t1 = (
            schema.get(trip_id=trip_id)
            .infer(["trip_id"], hr)
            .infer(["trip_id"], destination)
        )
        # mutate is a way to create new inferences.
        # here we are creating a new inference from destination --> is_tourist that maps zoo to true and everything
        # else to false
        t2 = t1.mutate(is_tourist=t1["destination"] == "Zoo").hide("destination")
        self.assertExpectedInline(
            str(t2),
            """\
[trip_id || hr is_tourist]
         hr  is_tourist
trip_id                
1         7        True
3         8        True
4         8        True
14       17        True
15       17        True
2         7       False
5         8       False
7        10       False
6         9       False
12       15       False
13       16       False
8        11       False
9        12       False
19       19       False
10       13       False
11       14       False
16       17       False
17       17       False
18       18       False
20       20       False

""",
        )

    def test_zoo_3(self):
        schema, trip_id, hr, destination = self.initialise()
        t1 = (
            schema.get(trip_id=trip_id)
            .infer(["trip_id"], hr)
            .infer(["trip_id"], destination)
        )
        t2 = t1.mutate(is_tourist=t1["destination"] == "Zoo").hide("destination")
        # group by changes the derivation information. Now we are inferring trip_id from hr and is_tourist,
        # this is basically like an inversion.
        t3 = t2.group_by(["hr", "is_tourist"])
        self.assertExpectedInline(
            str(t3),
            """\
[hr is_tourist || trip_id]
                    trip_id
hr is_tourist              
7  True               [1.0]
8  True          [3.0, 4.0]
17 True        [14.0, 15.0]
7  False              [2.0]
8  False              [5.0]
10 False              [7.0]
9  False              [6.0]
15 False             [12.0]
16 False             [13.0]
11 False              [8.0]
12 False              [9.0]
19 False             [19.0]
13 False             [10.0]
14 False             [11.0]
17 False       [16.0, 17.0]
18 False             [18.0]
20 False             [20.0]
263 keys hidden

""",
        )

    def test_zoo_4(self):
        schema, trip_id, hr, destination = self.initialise()
        t1 = (
            schema.get(trip_id=trip_id)
            .infer(["trip_id"], hr)
            .infer(["trip_id"], destination)
        )
        t2 = t1.mutate(is_tourist=t1["destination"] == "Zoo").hide("destination")
        t3 = t2.group_by(["hr", "is_tourist"])
        # mutate once again adds an inference
        t4 = (
            t3.mutate(num_trips=t3["trip_id"].count())
            .hide("trip_id")
            .sort(["hr", "is_tourist"])
        )
        self.assertExpectedInline(
            str(t4),
            """\
[hr is_tourist || num_trips]
               num_trips
hr is_tourist           
7  False             1.0
   True              1.0
8  False             1.0
   True              2.0
9  False             1.0
10 False             1.0
11 False             1.0
12 False             1.0
13 False             1.0
14 False             1.0
15 False             1.0
16 False             1.0
17 False             2.0
   True              2.0
18 False             1.0
19 False             1.0
20 False             1.0
11 keys hidden

""",
        )
