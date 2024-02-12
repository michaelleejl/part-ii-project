import expecttest

from schema import (
    compose_cardinality,
    Cardinality,
    get_indices_of_sublist,
    is_sublist,
    AtomicNode,
    SchemaEdge,
)
from schema.helpers.find_index import find_index
from schema.helpers.get_indices_of_sublist import SubListMustBeFullyContainedInList
from schema.helpers.invert_representation import invert_representation
from schema.helpers.list_difference import list_difference


class TestSchemaHelpers(expecttest.TestCase):

    def test_compose_cardinality_treats11AsIdentity(self):
        self.assertEqual(
            compose_cardinality(Cardinality.ONE_TO_ONE, Cardinality.ONE_TO_ONE),
            Cardinality.ONE_TO_ONE,
        )

        self.assertEqual(
            compose_cardinality(Cardinality.MANY_TO_ONE, Cardinality.ONE_TO_ONE),
            Cardinality.MANY_TO_ONE,
        )
        self.assertEqual(
            compose_cardinality(Cardinality.MANY_TO_ONE, Cardinality.MANY_TO_ONE),
            Cardinality.MANY_TO_ONE,
        )

        self.assertEqual(
            compose_cardinality(Cardinality.ONE_TO_MANY, Cardinality.ONE_TO_ONE),
            Cardinality.ONE_TO_MANY,
        )
        self.assertEqual(
            compose_cardinality(Cardinality.ONE_TO_ONE, Cardinality.ONE_TO_MANY),
            Cardinality.ONE_TO_MANY,
        )

        self.assertEqual(
            compose_cardinality(Cardinality.MANY_TO_MANY, Cardinality.ONE_TO_ONE),
            Cardinality.MANY_TO_MANY,
        )
        self.assertEqual(
            compose_cardinality(Cardinality.ONE_TO_ONE, Cardinality.MANY_TO_MANY),
            Cardinality.MANY_TO_MANY,
        )

    def test_compose_cardinality_correctlyComposesCardinalityN1(self):
        self.assertEqual(
            compose_cardinality(Cardinality.MANY_TO_ONE, Cardinality.MANY_TO_ONE),
            Cardinality.MANY_TO_ONE,
        )
        self.assertEqual(
            compose_cardinality(Cardinality.MANY_TO_ONE, Cardinality.ONE_TO_MANY),
            Cardinality.MANY_TO_MANY,
        )
        self.assertEqual(
            compose_cardinality(Cardinality.ONE_TO_MANY, Cardinality.MANY_TO_ONE),
            Cardinality.MANY_TO_MANY,
        )
        self.assertEqual(
            compose_cardinality(Cardinality.MANY_TO_ONE, Cardinality.MANY_TO_MANY),
            Cardinality.MANY_TO_MANY,
        )
        self.assertEqual(
            compose_cardinality(Cardinality.MANY_TO_MANY, Cardinality.MANY_TO_ONE),
            Cardinality.MANY_TO_MANY,
        )

    def test_compose_cardinality_correctlyComposesCardinality1N(self):
        self.assertEqual(
            compose_cardinality(Cardinality.ONE_TO_MANY, Cardinality.ONE_TO_MANY),
            Cardinality.ONE_TO_MANY,
        )
        self.assertEqual(
            compose_cardinality(Cardinality.ONE_TO_MANY, Cardinality.MANY_TO_MANY),
            Cardinality.MANY_TO_MANY,
        )
        self.assertEqual(
            compose_cardinality(Cardinality.MANY_TO_MANY, Cardinality.ONE_TO_MANY),
            Cardinality.MANY_TO_MANY,
        )

    def test_compose_cardinality_correctlyComposesCardinalityMN(self):
        self.assertEqual(
            compose_cardinality(Cardinality.MANY_TO_MANY, Cardinality.MANY_TO_MANY),
            Cardinality.MANY_TO_MANY,
        )

    def test_find_index_successfullyReturnsIndexIfItemInList(self):
        self.assertEqual(find_index(2, [1, 2, 3]), 1)

    def test_find_index_returnsMinusOneIfItemNotInList(self):
        self.assertEqual(find_index(4, [1, 2, 3]), -1)

    def test_find_index_returnsMinusOneIfListIsEmpty(self):
        self.assertEqual(find_index(4, []), -1)

    def test_get_indices_of_sublist_successfullyReturnsIndices(self):
        self.assertEqual(get_indices_of_sublist([1, 3], [1, 2, 3]), [0, 2])

    def test_get_indices_of_sublist_returnsEmptyListIfSublistIsEmpty(self):
        self.assertEqual(get_indices_of_sublist([], [1, 2, 3]), [])

    def test_get_indices_of_sublist_raisesExceptionIfSublistNotFullyContainedInList(
        self,
    ):
        self.assertExpectedRaisesInline(
            SubListMustBeFullyContainedInList,
            lambda: get_indices_of_sublist([1, 4], [1, 2, 3]),
            """Sublist [1, 4] is not fully contained in list [1, 2, 3]""",
        )
        self.assertExpectedRaisesInline(
            SubListMustBeFullyContainedInList,
            lambda: get_indices_of_sublist([1, 2, 3, 4], [1, 2, 3]),
            """Sublist [1, 2, 3, 4] is not fully contained in list [1, 2, 3]""",
        )

    def test_invert_representation_steps_successfullyInvertsSeriesOfSteps(self):
        from representation.representation import Traverse, StartTraversal, EndTraversal
        from frontend.domain import Domain

        u = AtomicNode("u")
        u.id_prefix = 0
        v = AtomicNode("v")
        v.id_prefix = 0
        w = AtomicNode("w")
        w.id_prefix = 0

        e1 = SchemaEdge(u, v, Cardinality.MANY_TO_ONE)
        e2 = SchemaEdge(v, w, Cardinality.MANY_TO_ONE)

        start_columns = [Domain("start", u)]
        end_columns = [Domain("end", w)]
        stt = StartTraversal(start_columns)
        trv1 = Traverse(e1)
        trv2 = Traverse(e2)
        ent = EndTraversal(end_columns)
        self.assertExpectedInline(
            str(invert_representation([stt, trv1, trv2, ent])),
            """[STT <[end]>, TRV <w <--- v, [v]>, TRV <v <--- u, [u]>, ENT <[end], [start]>]"""
            "",
        )

    def test_is_sublist_returnsTrueIfList1IsSublistOfList2(self):
        self.assertTrue(is_sublist([], [1, 2, 3]))
        self.assertTrue(is_sublist([1, 3], [1, 2, 3]))

    def test_is_sublist_returnsFalseIfList1IsNotSublistOfList2(self):
        self.assertFalse(is_sublist([1, 2, 3, 4], [1, 2, 3]))
        self.assertFalse(is_sublist([1, 2, 4], [1, 2, 3]))

    def test_list_difference_successfullyReturnsListDifference(self):
        self.assertEqual(list_difference([1, 2, 3], [2]), [1, 3])
