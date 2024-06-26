import expecttest
import pandas as pd

from schema.schema import Schema


class TestEx8(expecttest.TestCase):

    def initialise(self):
        s = Schema()

        persons_df = pd.read_csv("csv/pets/persons.csv").set_index("person")
        allergies_df = pd.read_csv("./csv/pets/allergies.csv").set_index(
            ["person", "pet"]
        )

        persons = s.insert_dataframe(persons_df)
        allergies = s.insert_dataframe(allergies_df)

        p_person = persons["person"]
        a_person = allergies["person"]

        p_pet = persons["pet"]
        a_pet = allergies["pet"]

        Person = s.create_class("Person")
        Pet = s.create_class("Pet")

        s.blend(p_person, a_person, Person)
        s.blend(p_pet, a_pet, Pet)
        return s, allergies, persons, Person, Pet

        # Schema:
        # person -> age
        # person -> pet
        # person, pet -> allergy

    # GOAL 1: [person pet || age]
    def test_ex8_goal1_step1_getThenInfer(self):
        s, allergies, persons, Person, Pet = self.initialise()
        t1 = (
            s.get(person=persons["person"], pet=persons["pet"])
            .infer(["person"], persons["age"])
            .sort(["person", "pet"])
        )
        self.assertExpectedInline(
            str(t1),
            """\
[person pet || age]
               age
person pet        
George Buster   23
       Corky    23
       Martha   23
       Pepper   23
John   Buster   27
       Corky    27
       Martha   27
       Pepper   27
Paul   Buster   25
       Corky    25
       Martha   25
       Pepper   25
Ringo  Buster   29
       Corky    29
       Martha   29
       Pepper   29

""",
        )
        # [persons.person persons.pet || person.age]
        # John            Pepper      || 27
        # John            Martha      || 27
        # John            Corky       || 27
        # John            Buster      || 27
        # ...

    def test_ex8_goal2_step1_getThenInfer(self):
        s, allergies, persons, Person, Pet = self.initialise()
        t11 = (
            s.get(person=persons["person"])
            .infer(["person"], persons["pet"])
            .infer(["person"], persons["age"])
        )
        self.assertExpectedInline(
            str(t11),
            """\
[person || pet age]
           pet  age
person             
John    Pepper   27
Paul    Martha   25
George   Corky   23
Ringo   Buster   29

""",
        )
        # [persons.person || persons.pet  persons.age]
        #  John           || Pepper       27
        #  Paul           || Martha       25
        #  George         || Corky        23
        #  Ringo          || Buster       29

    def test_ex8_goal2_step2_setKey(self):
        s, allergies, persons, Person, Pet = self.initialise()
        t11 = (
            s.get(person=persons["person"])
            .infer(["person"], persons["pet"])
            .infer(["person"], persons["age"])
        )
        t12 = t11.shift_right()
        self.assertExpectedInline(
            str(t12),
            """\
[person pet || age]
               age
person pet        
John   Pepper   27
Paul   Martha   25
George Corky    23
Ringo  Buster   29
12 keys hidden

""",
        )
        # [persons.person persons.pet || persons.age]
        #  John           Pepper      ||  27
        #  Paul           Martha      ||  25
        #  George         Corky       ||  23
        #  Ringo          Buster      ||  29

        # Note that these methods are semantically different!

    # CAN I CONVERT BETWEEN THE TWO?
    def test_ex8_conversion1_step1_infer(self):
        s, allergies, persons, Person, Pet = self.initialise()
        t1 = s.get(person=persons["person"], pet=persons["pet"]).infer(
            ["person"], persons["age"]
        )
        t2 = t1.infer(["person"], persons["pet"]).sort(["person", "pet"])
        self.assertExpectedInline(
            str(t2),
            """\
[person pet || age pet_1]
               age   pet_1
person pet                
George Buster   23   Corky
       Corky    23   Corky
       Martha   23   Corky
       Pepper   23   Corky
John   Buster   27  Pepper
       Corky    27  Pepper
       Martha   27  Pepper
       Pepper   27  Pepper
Paul   Buster   25  Martha
       Corky    25  Martha
       Martha   25  Martha
       Pepper   25  Martha
Ringo  Buster   29  Buster
       Corky    29  Buster
       Martha   29  Buster
       Pepper   29  Buster

""",
        )
        # [persons.person persons.pet || person.age  persons.pet (1)]
        # John            Pepper      || 27          Pepper
        # John            Martha      || 27          Pepper
        # John            Corky       || 27          Pepper
        # John            Buster      || 27          Pepper
        # ...
        # note the automatic renaming to keep names distinct

    def test_ex8_conversion1_step2_setKey(self):
        s, allergies, persons, Person, Pet = self.initialise()
        t1 = s.get(person=persons["person"], pet=persons["pet"]).infer(
            ["person"], persons["age"]
        )
        t2 = t1.infer(["person"], persons["pet"])
        t3 = t2.swap("age", "pet_1")
        t4 = t3.shift_right().sort(["person", "pet", "pet_1"])
        self.assertExpectedInline(
            str(t4),
            """\
[person pet pet_1 || age]
                      age
person pet    pet_1      
George Buster Corky    23
       Corky  Corky    23
       Martha Corky    23
       Pepper Corky    23
John   Buster Pepper   27
       Corky  Pepper   27
       Martha Pepper   27
       Pepper Pepper   27
Paul   Buster Martha   25
       Corky  Martha   25
       Martha Martha   25
       Pepper Martha   25
Ringo  Buster Buster   29
       Corky  Buster   29
       Martha Buster   29
       Pepper Buster   29
48 keys hidden

""",
        )
        # [persons.person persons.pet  persons.pet (1)  || persons.age]
        # John            Pepper       Pepper           || 27
        # John            Martha       Pepper           || 27
        # John            Corky        Pepper           || 27
        # John            Buster       Pepper           || 27
        # ...

    # how to throw away the old key?
    # I can equate
    def test_ex8_conversion1_step3_equate(self):
        s, allergies, persons, Person, Pet = self.initialise()
        t1 = s.get(person=persons["person"], pet=persons["pet"]).infer(
            ["person"], persons["age"]
        )
        t2 = t1.infer(["person"], persons["pet"])
        t3 = t2.swap("age", "pet_1")
        t4 = t3.shift_right()
        t5 = t4.mutate(is_own_pet=t4["pet"].mask(t4["pet"] == t4["pet_1"])).filter(
            "is_own_pet"
        )
        self.maxDiff = None
        self.assertExpectedInline(
            str(t5),
            """\
[person pet pet_1 || age is_own_pet]
                      age is_own_pet
person pet    pet_1                 
John   Pepper Pepper   27     Pepper
Paul   Martha Martha   25     Martha
George Corky  Corky    23      Corky
Ringo  Buster Buster   29     Buster
60 keys hidden

""",
        )

        # [persons.person persons.pet || persons.age]
        #  John           Pepper      ||  27
        #  Paul           Martha      ||  25
        #  George         Corky       ||  23
        #  Ringo          Buster      ||  29

        # OR, since persons.pet and persons.pet (1) are both weak keys for persons.age
        # and equating is only different from filter + hide in that it takes weak keys and makes them strong
        # in this case there is no difference. PLUS we don't even need to do a filter.

    # GOAL 2: [person pet || age allergy]
    def test_ex8_goal3_step1_get(self):
        s, allergies, persons, Person, Pet = self.initialise()
        t21 = s.get(Person=Person, Pet=Pet).sort(["Person", "Pet"])
        self.maxDiff = None
        self.assertExpectedInline(
            str(t21),
            """\
[Person Pet || ]
Empty DataFrame
Columns: []
Index: []
30 keys hidden

""",
        )
        # [Person Pet     || ]
        #  John   Pepper
        #  John   Martha
        #  John   Corky
        #  John   Buster
        #  John   Strawberry
        #  John   Fido
        #  Paul   Pepper
        # ...

    def test_ex8_goal3_step2_infer(self):
        s, allergies, persons, Person, Pet = self.initialise()
        t21 = s.get(Person=Person, Pet=Pet)
        t22 = t21.infer(["Person", "Pet"], allergies["allergy"])
        self.assertExpectedInline(
            str(t22),
            """\
[Person Pet || allergy]
                      allergy
Person Pet                   
Paul   Martha           Sheep
       Strawberry  Chocolates
Pete   Fido               Hay
George Corky             Dust
26 keys hidden

""",
        )
        # [Person Pet         || allergies.allergy ]
        #  Paul   Martha      || Sheep
        #  Paul   Strawberry  || Chocolates
        #  Pete   Fido        || Hay
        #  George Corky       || Dust

    def test_ex8_goal3_step3_infer(self):
        s, allergies, persons, Person, Pet = self.initialise()
        t21 = s.get(Person=Person, Pet=Pet)
        t22 = t21.infer(["Person", "Pet"], allergies["allergy"])
        t23 = t22.infer(["Person"], persons["age"]).sort(["Person", "Pet"])
        self.maxDiff = None
        self.assertExpectedInline(
            str(t23),
            """\
[Person Pet || allergy age]
                      allergy   age
Person Pet                         
George Buster             NaN  23.0
       Corky             Dust  23.0
       Fido               NaN  23.0
       Martha             NaN  23.0
       Pepper             NaN  23.0
       Strawberry         NaN  23.0
John   Buster             NaN  27.0
       Corky              NaN  27.0
       Fido               NaN  27.0
       Martha             NaN  27.0
       Pepper             NaN  27.0
       Strawberry         NaN  27.0
Paul   Buster             NaN  25.0
       Corky              NaN  25.0
       Fido               NaN  25.0
       Martha           Sheep  25.0
       Pepper             NaN  25.0
       Strawberry  Chocolates  25.0
Pete   Fido               Hay   NaN
Ringo  Buster             NaN  29.0
       Corky              NaN  29.0
       Fido               NaN  29.0
       Martha             NaN  29.0
       Pepper             NaN  29.0
       Strawberry         NaN  29.0
5 keys hidden

""",
        )
        # [Person Pet         || allergies.allergy   persons.age]
        #  John   Pepper      || NA                  27
        #  John   Martha      || NA                  27
        #  John   Corky       || NA                  27
        #  John   Buster      || NA                  27
        #  John   Strawberry  || NA                  27
        #  John   Fido        || NA                  27
        #  Paul   Pepper      || NA                  25
        #  Paul   Martha      || Sheep               25
        # ...
