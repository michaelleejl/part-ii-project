import unittest
import pandas as pd

from schema.exceptions import KeyDuplicationException
from backend.helpers.check_for_duplicate_keys import check_for_duplicate_keys


class TestCheckForDuplicateKeys(unittest.TestCase):
    def test_check_for_duplicate_keys_raisesExceptionIfThereAreDuplicateKeys(self):
        keys_with_duplication = pd.DataFrame([0, 0, 1])
        self.assertRaises(KeyDuplicationException, lambda: check_for_duplicate_keys(keys_with_duplication))

    def test_check_for_duplicate_keys_doesNotRaiseExceptionsIfThereAreNoDuplicateKeys(self):
        keys_without_duplication = pd.DataFrame([0, 1, 2])
        check_for_duplicate_keys(keys_without_duplication)

