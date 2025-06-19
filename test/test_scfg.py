import random
import unittest

from formal_gym.scfg import SCFG


class TestSCFG(unittest.TestCase):
    def setUp(self):
        sample_str = """
        S -> <'a', 'b'>
        S -> <A B, B A>
        A -> <'x', 'y'>
        B -> <'z', 'w'>
        """
        self.scfg = SCFG(sync_str=sample_str)  # type: ignore
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
        self.scfg = SCFG(sync_str=null_scfg_str)
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
