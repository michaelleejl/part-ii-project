import expecttest
import pandas as pd

from schema import Schema, SchemaNode


class TestEx8(expecttest.TestCase):

    def initialise(self) -> Schema:
        s = Schema()

        persons = pd.read_csv("csv/pets/persons.csv").set_index("person")
        allergies = pd.read_csv("./csv/pets/allergies.csv").set_index(["person", "pet"])

        s.insert_dataframe(persons, "persons")
        s.insert_dataframe(allergies, "allergies")

        p_person = SchemaNode("person", cluster="persons")
        a_person = SchemaNode("person", cluster="allergies")

        p_pet = SchemaNode("pet", cluster="persons")
        a_pet = SchemaNode("pet", cluster="allergies")

        s.blend(p_person, a_person, under="Person")
        s.blend(p_pet, a_pet, under="Pet")
        return s

        # Schema:
        # person -> age
        # person -> pet
        # person, pet -> allergy

    # GOAL 1: [person pet || age]
    def test_ex8_goal1_step1_getThenInfer(self):
        s = self.initialise()
        t1 = s.get(["persons.person", "persons.pet"]).infer(["persons.person"], "persons.age")
        self.assertExpectedInline(str(t1), """\
[persons.person persons.pet || persons.age]
                            persons.age
persons.person persons.pet             
John           Pepper                27
               Martha                27
               Corky                 27
               Buster                27
Paul           Pepper                25
               Martha                25
               Corky                 25
               Buster                25
George         Pepper                23
               Martha                23
               Corky                 23
               Buster                23
Ringo          Pepper                29
               Martha                29
               Corky                 29
               Buster                29

""")
        # [persons.person persons.pet || person.age]
        # John            Pepper      || 27
        # John            Martha      || 27
        # John            Corky       || 27
        # John            Buster      || 27
        # ...

    def test_ex8_goal2_step1_getThenInfer(self):
        s = self.initialise()
        t11 = s.get(["persons.person"]).infer(["persons.person"], "persons.pet").infer(["persons.person"], "persons.age")
        self.assertExpectedInline(str(t11), """\
[persons.person || persons.pet persons.age]
               persons.pet  persons.age
persons.person                         
John                Pepper           27
Paul                Martha           25
George               Corky           23
Ringo               Buster           29

""")
        # [persons.person || persons.pet  persons.age]
        #  John           || Pepper       27
        #  Paul           || Martha       25
        #  George         || Corky        23
        #  Ringo          || Buster       29

    def test_ex8_goal2_step2_setKey(self):
        s = self.initialise()
        t11 = s.get(["persons.person"]).infer(["persons.person"], "persons.pet").infer(["persons.person"], "persons.age")
        t12 = t11.set_key(["persons.person", "persons.pet"])
        self.assertExpectedInline(str(t12), """\
[persons.person persons.pet || persons.age]
                            persons.age
persons.person persons.pet             
John           Pepper                27
Paul           Martha                25
George         Corky                 23
Ringo          Buster                29

""")
        # [persons.person persons.pet || persons.age]
        #  John           Pepper      ||  27
        #  Paul           Martha      ||  25
        #  George         Corky       ||  23
        #  Ringo          Buster      ||  29

        # Note that these methods are semantically different!

# CAN I CONVERT BETWEEN THE TWO?
    def test_ex8_conversion1_step1_infer(self):
        s = self.initialise()
        t1 = s.get(["persons.person", "persons.pet"]).infer(["persons.person"], "persons.age")
        t2 = t1.infer(["persons.person"], "persons.pet")
        self.assertExpectedInline(str(t2), """\
[persons.person persons.pet || persons.age persons.pet_1]
                            persons.age persons.pet_1
persons.person persons.pet                           
John           Pepper                27        Pepper
               Martha                27        Pepper
               Corky                 27        Pepper
               Buster                27        Pepper
Paul           Pepper                25        Martha
               Martha                25        Martha
               Corky                 25        Martha
               Buster                25        Martha
George         Pepper                23         Corky
               Martha                23         Corky
               Corky                 23         Corky
               Buster                23         Corky
Ringo          Pepper                29        Buster
               Martha                29        Buster
               Corky                 29        Buster
               Buster                29        Buster

""")
        # [persons.person persons.pet || person.age  persons.pet (1)]
        # John            Pepper      || 27          Pepper
        # John            Martha      || 27          Pepper
        # John            Corky       || 27          Pepper
        # John            Buster      || 27          Pepper
        # ...
        # note the automatic renaming to keep names distinct

    def test_ex8_conversion1_step2_setKey(self):
        s = self.initialise()
        t1 = s.get(["persons.person", "persons.pet"]).infer(["persons.person"], "persons.age")
        t2 = t1.infer(["persons.person"], "persons.pet")
        t3 = t2.set_key(["persons.person", "persons.pet", "persons.pet_1"])
        self.assertExpectedInline(str(t3), """\
[persons.person persons.pet persons.pet_1 || persons.age]
                                          persons.age
persons.person persons.pet persons.pet_1             
John           Pepper      Pepper                  27
               Martha      Pepper                  27
               Corky       Pepper                  27
               Buster      Pepper                  27
Paul           Pepper      Martha                  25
               Martha      Martha                  25
               Corky       Martha                  25
               Buster      Martha                  25
George         Pepper      Corky                   23
               Martha      Corky                   23
               Corky       Corky                   23
               Buster      Corky                   23
Ringo          Pepper      Buster                  29
               Martha      Buster                  29
               Corky       Buster                  29
               Buster      Buster                  29

""")
        # [persons.person persons.pet  persons.pet (1)  || persons.age]
        # John            Pepper       Pepper           || 27
        # John            Martha       Pepper           || 27
        # John            Corky        Pepper           || 27
        # John            Buster       Pepper           || 27
        # ...

    # how to throw away the old key?
    # I can equate
    def test_ex8_conversion1_step3_equate(self):
        s = self.initialise()
        t1 = s.get(["persons.person", "persons.pet"]).infer(["persons.person"], "persons.age")
        t2 = t1.infer(["persons.person"], "persons.pet")
        t3 = t2.set_key(["persons.person", "persons.pet", "persons.pet_1"])
        t4 = t3.equate("persons.pet", "persons.pet_1")
        self.assertExpectedInline(str(t4), """\
[persons.person persons.pet || persons.age]
                            persons.age
persons.person persons.pet             
John           Pepper                27
Paul           Martha                25
George         Corky                 23
Ringo          Buster                29

""")

        # [persons.person persons.pet || persons.age]
        #  John           Pepper      ||  27
        #  Paul           Martha      ||  25
        #  George         Corky       ||  23
        #  Ringo          Buster      ||  29

        # OR, since persons.pet and persons.pet (1) are both weak keys for persons.age
        # and equating is only different from filter + hide in that it takes weak keys and makes them strong
        # in this case there is no difference. PLUS we don't even need to do a filter.

    def test_ex8_conversion1_step4_hide(self):
        s = self.initialise()
        t1 = s.get(["persons.person", "persons.pet"]).infer(["persons.person"], "persons.age")
        t2 = t1.infer(["persons.person"], "persons.pet")
        t3 = t2.set_key(["persons.person", "persons.pet", "persons.pet_1"])
        t4 = t3.equate("persons.pet", "persons.pet_1")
        t5 = t4.hide("persons.pet")
        self.assertExpectedInline(str(t5), """\
[persons.person || persons.age]
                persons.age
persons.person             
John                     27
Paul                     25
George                   23
Ringo                    29

""")
        # [persons.person persons.pet (1) || persons.age]
        #  John           Pepper          ||  27
        #  Paul           Martha          ||  25
        #  George         Corky           ||  23
        #  Ringo          Buster          ||  29

        # # Can I convert in the other direction? Yes
        # # Call compose

    def test_ex8_conversion2_step1_compose(self):
        s = self.initialise()
        t11 = s.get(["persons.person"]).infer(["persons.person"], "persons.pet").infer(["persons.person"],
                                                                                       "persons.age")
        t12 = t11.set_key(["persons.person", "persons.pet"])
        t13 = t12.compose(["persons.person", "persons.pet"], "persons.person")
        self.maxDiff = None
        self.assertExpectedInline(str(t13), """\
[persons.person persons.pet_1 persons.pet || persons.age]
                                          persons.age
persons.person persons.pet_1 persons.pet             
John           Pepper        Pepper                27
               Martha        Pepper                27
               Corky         Pepper                27
               Buster        Pepper                27
Paul           Pepper        Martha                25
               Martha        Martha                25
               Corky         Martha                25
               Buster        Martha                25
George         Pepper        Corky                 23
               Martha        Corky                 23
               Corky         Corky                 23
               Buster        Corky                 23
Ringo          Pepper        Buster                29
               Martha        Buster                29
               Corky         Buster                29
               Buster        Buster                29

""")
        # [persons.person persons.pet  persons.pet (1)  || persons.age]
        # John            Pepper       Pepper           || 27
        # John            Martha       Pepper           || 27
        # John            Corky        Pepper           || 27
        # John            Buster       Pepper           || 27
        # ...
        # I can call compose because there is a path from person x pet x pet' -> person x pet (projection)

    def test_ex8_conversion2_step2_hide(self):
        s = self.initialise()
        t11 = s.get(["persons.person"]).infer(["persons.person"], "persons.pet").infer(["persons.person"],
                                                                                       "persons.age")
        t12 = t11.set_key(["persons.person", "persons.pet"])
        t13 = t12.compose(["persons.person", "persons.pet"], "persons.person")
        t14 = t13.hide("persons.pet")
        self.assertExpectedInline(str(t14), """\
[persons.person persons.pet_1 || persons.age]
                              persons.age
persons.person persons.pet_1             
John           Pepper                  27
               Martha                  27
               Corky                   27
               Buster                  27
Paul           Pepper                  25
               Martha                  25
               Corky                   25
               Buster                  25
George         Pepper                  23
               Martha                  23
               Corky                   23
               Buster                  23
Ringo          Pepper                  29
               Martha                  29
               Corky                   29
               Buster                  29

""")

# GOAL 2: [person pet || age allergy]
    def test_ex8_goal3_step1_get(self):
        s = self.initialise()
        t21 = s.get(["Person", "Pet"])
        self.assertExpectedInline(str(t21), """\
[Person Pet || ]
Empty DataFrame
Columns: []
Index: [(John, Pepper), (John, Martha), (John, Corky), (John, Buster), (John, Strawberry), (John, Fido), (Paul, Pepper), (Paul, Martha), (Paul, Corky), (Paul, Buster), (Paul, Strawberry), (Paul, Fido), (George, Pepper), (George, Martha), (George, Corky), (George, Buster), (George, Strawberry), (George, Fido), (Ringo, Pepper), (Ringo, Martha), (Ringo, Corky), (Ringo, Buster), (Ringo, Strawberry), (Ringo, Fido), (Pete, Pepper), (Pete, Martha), (Pete, Corky), (Pete, Buster), (Pete, Strawberry), (Pete, Fido)]

""")
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
        s = self.initialise()
        t21 = s.get(["Person", "Pet"])
        t22 = t21.infer(["Person", "Pet"], "allergies.allergy")
        self.assertExpectedInline(str(t22), """\
[Person Pet || allergies.allergy]
                  allergies.allergy
Person Pet                         
Paul   Martha                 Sheep
       Strawberry        Chocolates
Pete   Fido                     Hay
George Corky                   Dust
26 keys hidden

""")
        # [Person Pet         || allergies.allergy ]
        #  Paul   Martha      || Sheep
        #  Paul   Strawberry  || Chocolates
        #  Pete   Fido        || Hay
        #  George Corky       || Dust

    def test_ex8_goal3_step3_infer(self):
        s = self.initialise()
        t21 = s.get(["Person", "Pet"])
        t22 = t21.infer(["Person", "Pet"], "allergies.allergy")
        t23 = t22.infer(["Person"], "persons.age").sort(["Person", "Pet"])
        self.maxDiff = None
        self.assertExpectedInline(str(t23), """\
[Person Pet || allergies.allergy persons.age]
                  allergies.allergy  persons.age
Person Pet                                      
George Buster                   NaN         23.0
       Corky                   Dust         23.0
       Fido                     NaN         23.0
       Martha                   NaN         23.0
       Pepper                   NaN         23.0
       Strawberry               NaN         23.0
John   Buster                   NaN         27.0
       Corky                    NaN         27.0
       Fido                     NaN         27.0
       Martha                   NaN         27.0
       Pepper                   NaN         27.0
       Strawberry               NaN         27.0
Paul   Buster                   NaN         25.0
       Corky                    NaN         25.0
       Fido                     NaN         25.0
       Martha                 Sheep         25.0
       Pepper                   NaN         25.0
       Strawberry        Chocolates         25.0
Pete   Fido                     Hay          NaN
Ringo  Buster                   NaN         29.0
       Corky                    NaN         29.0
       Fido                     NaN         29.0
       Martha                   NaN         29.0
       Pepper                   NaN         29.0
       Strawberry               NaN         29.0
5 keys hidden

""")
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