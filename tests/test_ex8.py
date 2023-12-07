import expecttest
import pandas as pd

from schema import Schema, SchemaNode


class TestEx8(expecttest.TestCase):
    def test_ex_8(self):
        s = Schema()

        persons = pd.read_csv("csv/pets/persons.csv").set_index("person")
        allergies = pd.read_csv("./csv/pets/allergies.csv").set_index(["person", "pet"])

        s.insert_dataframe(persons, "persons")
        s.insert_dataframe(allergies, "allergies")

        p_person = SchemaNode("person", cluster="persons")
        a_person = SchemaNode("person", cluster="allergies")

        p_pet = SchemaNode("pet", cluster="persons")
        a_pet = SchemaNode("pet", cluster="allergies")

        s.blend(p_person, a_person, under ="Person")
        s.blend(p_pet, a_pet, under = "Pet")

        # ========================================================================
        # ========================================================================
        # Schema:
        # person -> age
        # person -> pet
        # person, pet -> allergy

        # GOAL 1: [person pet || age]
        # I could do
        p_age = SchemaNode("age", cluster="persons")

        # WAY 1
        t1 = s.get([p_person, p_pet]).infer([p_person], p_age)
        # [persons.person persons.pet || person.age]
        # John            Pepper      || 27
        # John            Martha      || 27
        # John            Corky       || 27
        # John            Buster      || 27
        # ...

        # WAY 2
        # Or I could do
        t11 = s.get([p_person]).infer([p_person], p_pet).infer([p_person], p_age)
        # [persons.person || persons.pet  persons.age]
        #  John           || Pepper       27
        #  Paul           || Martha       25
        #  George         || Corky        23
        #  Ringo          || Buster       29

        t12 = t11.set_key(["persons.person", "persons.pet"])
        # [persons.person persons.pet || persons.age]
        #  John           Pepper      ||  27
        #  Paul           Martha      ||  25
        #  George         Corky       ||  23
        #  Ringo          Buster      ||  29

        # Note that these methods are semantically different!

        # CAN I CONVERT BETWEEN THE TWO?
        # To convert the first into the second
        # The idea is to introduce the constraint
        # and then "throw away" the old key
        t2 = t1.infer([p_person], p_pet)
        # [persons.person persons.pet || person.age  persons.pet (1)]
        # John            Pepper      || 27          Pepper
        # John            Martha      || 27          Pepper
        # John            Corky       || 27          Pepper
        # John            Buster      || 27          Pepper
        # ...
        # note the automatic renaming to keep names distinct

        t2 = t1.set_key([p_person, p_pet, SchemaNode("pet (1)", cluster="persons")])
        # [persons.person persons.pet  persons.pet (1)  || persons.age]
        # John            Pepper       Pepper           || 27
        # John            Martha       Pepper           || 27
        # John            Corky        Pepper           || 27
        # John            Buster       Pepper           || 27
        # ...

        # how to throw away the old key?
        # I can equate
        t3 = t2.equate("persons.pet", "persons.pet (1)")
        # [persons.person persons.pet || persons.age]
        #  John           Pepper      ||  27
        #  Paul           Martha      ||  25
        #  George         Corky       ||  23
        #  Ringo          Buster      ||  29

        # OR, since persons.pet and persons.pet (1) are both weak keys for persons.age
        # and equating is only different from filter + hide in that it takes weak keys and makes them strong
        # in this case there is no difference. PLUS we don't even need to do a filter.

        t3 = t2.hide(["persons.pet"])
        # [persons.person persons.pet (1) || persons.age]
        #  John           Pepper          ||  27
        #  Paul           Martha          ||  25
        #  George         Corky           ||  23
        #  Ringo          Buster          ||  29

        # Can I convert in the other direction? Yes
        # Call compose
        t13 = t12.compose(["persons.person", "persons.pet", p_pet], ["persons.person", "persons.pet"])
        # [persons.person persons.pet  persons.pet (1)  || persons.age]
        # John            Pepper       Pepper           || 27
        # John            Martha       Pepper           || 27
        # John            Corky        Pepper           || 27
        # John            Buster       Pepper           || 27
        # ...
        # I can call compose because there is a path from person x pet x pet' -> person x pet (projection)
        t14 = t13.hide(["persons.pet"])


        # GOAL 2: [person pet || age allergy]
        t21 = s.get([SchemaNode("Person"), SchemaNode("Pet")])
        # [Person Pet     || ]
        #  John   Pepper
        #  John   Martha
        #  John   Corky
        #  John   Buster
        #  John   Strawberry
        #  John   Fido
        #  Paul   Pepper
        # ...

        t22 = t21.infer(["Person", "Pet"], SchemaNode("allergy", cluster="allergies"))
        # [Person Pet         || allergies.allergy ]
        #  Paul   Martha      || Sheep
        #  Paul   Strawberry  || Chocolates
        #  Pete   Fido        || Hay
        #  George Corky       || Dust

        t23 = t22.infer(["Person"], p_age)
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