import unittest

from formal_gym import metagrammar as mg


class TestSymbols(unittest.TestCase):
    def test_terminal_str(self):
        t = mg.Terminal("a")
        self.assertEqual(f"{t}", "'a'")

    def test_nonterminal_str(self):
        nt = mg.START_SYMBOL
        self.assertEqual(f"{nt}", "S")

    def test_production_str(self):
        p = mg.Production(mg.START_SYMBOL, (mg.Terminal("a"), mg.Terminal("b")))
        self.assertEqual(f"{p}", "S -> 'a' 'b'")

    def test_lexical_production_length(self):
        p = mg.Production(mg.START_SYMBOL, (mg.Terminal("a")))
        self.assertEqual(p.length, 1)

    def test_binary_production_length(self):
        p = mg.Production(
            mg.START_SYMBOL, (mg.Nonterminal("NT1"), mg.Nonterminal("NT2"))
        )
        self.assertEqual(p.length, 2)

    def test_symbol_type(self):
        t = mg.Terminal("a")
        nt = mg.START_SYMBOL

        self.assertTrue(isinstance(t, mg.Symbol))
        self.assertTrue(isinstance(t, mg.Terminal))
        self.assertFalse(isinstance(t, mg.Nonterminal))

        self.assertTrue(isinstance(nt, mg.Symbol))
        self.assertTrue(isinstance(nt, mg.Nonterminal))
        self.assertFalse(isinstance(nt, mg.Terminal))


def main():
    unittest.main()


if __name__ == "__main__":
    main()
