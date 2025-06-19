import random
import unittest
from unittest.mock import patch

from formal_gym.scfg import SCFG


# Patch SyncGrammarParams before defining the mock class
def mock_generate_scfg(sync_params):
    # Simple grammar: S -> <'a', 'b'> | <A, B>; A -> <'x', 'y'>; B -> <'z', 'w'>
    return """
    S -> <'a', 'b'>
    S -> <A B, B A>
    A -> <'x', 'y'>
    B -> <'z', 'w'>
    """


class TestSCFG(unittest.TestCase):
    def setUp(self):
        # Patch both generate_scfg and SyncGrammarParams for the duration of each test
        patcher1 = patch("formal_gym.scfg.fg_mxg.generate_scfg", mock_generate_scfg)
        patcher2 = patch(
            "formal_gym.scfg.fg_mxg.SyncGrammarParams",
            type("PatchedSyncGrammarParams", (), {}),
        )
        self.addCleanup(patcher1.stop)
        self.addCleanup(patcher2.stop)
        self.mock_generate = patcher1.start()
        PatchedSyncGrammarParams = patcher2.start()

        # Define a mock class that inherits from the patched SyncGrammarParams
        class MockSyncGrammarParams(PatchedSyncGrammarParams):
            pass

        self.MockSyncGrammarParams = MockSyncGrammarParams
        self.scfg = SCFG(MockSyncGrammarParams())  # type: ignore
        self.rng = random.Random(123)

    def test_parse_rules(self):
        rules = self.scfg.rules
        self.assertIn("S", rules)
        self.assertIn("A", rules)
        self.assertIn("B", rules)
        self.assertEqual(rules["S"][0], (("'a'",), ("'b'",)))
        self.assertEqual(rules["A"][0], (("'x'",), ("'y'",)))
        self.assertEqual(rules["B"][0], (("'z'",), ("'w'",)))

    def test_sample_terminal(self):
        # Force sampling the terminal rule
        self.rng.seed(0)
        sample: dict[str, str] = self.scfg.sample(max_depth=1, rng=self.rng)
        self.assertIn(sample["left"], ["a", "x z"])
        self.assertIn(sample["right"], ["b", "w y"])
        self.assertTrue(sample["left_tree"].startswith("(S"))
        self.assertTrue(sample["right_tree"].startswith("(S"))

    def test_phonetic_filter(self):
        # Define a new grammar that guarantees a null symbol is produced
        null_scfg_str = """
        S -> <'∅null', '∅null'>
        """
        self.scfg.rules = self.scfg._parse_rules(null_scfg_str)
        self.rng.seed(1)
        sample: dict[str, str] = self.scfg.sample(max_depth=1, rng=self.rng)

        self.assertEqual(sample["left"], "∅null")
        self.assertEqual(sample["right"], "∅null")
        self.assertEqual(sample["left_phonetic"], "")
        self.assertEqual(sample["right_phonetic"], "")
        self.assertEqual(sample["left_tree"], "(S ∅null)")
        self.assertEqual(sample["right_tree"], "(S ∅null)")


if __name__ == "__main__":
    unittest.main()
