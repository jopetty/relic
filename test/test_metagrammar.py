import unittest

from formal_gym import metagrammar as mg


class TestSymbols(unittest.TestCase):
    def test_terminal_str(self):
        t = mg.Terminal("t0")
        self.assertEqual(f"{t}", "'t0'")

    def test_nonterminal_str(self):
        nt = mg.START_SYMBOL
        self.assertEqual(f"{nt}", "S")

        nt1 = mg.Nonterminal("NT1")
        self.assertEqual(f"{nt1}", "NT1")

    def test_production_str(self):
        p = mg.Production(mg.START_SYMBOL, (mg.Terminal("t0"), mg.Terminal("t1")))
        self.assertEqual(f"{p}", "S -> 't0' 't1'")

    def test_lexical_production_length(self):
        p = mg.Production(mg.START_SYMBOL, (mg.Terminal("t0"),))
        self.assertEqual(p.length, 1)

    def test_binary_production_length(self):
        p = mg.Production(
            mg.START_SYMBOL, (mg.Nonterminal("NT1"), mg.Nonterminal("NT2"))
        )
        self.assertEqual(p.length, 2)

    def test_symbol_type(self):
        t = mg.Terminal("t0")
        nt = mg.START_SYMBOL

        self.assertTrue(isinstance(t, mg.Symbol))
        self.assertTrue(isinstance(t, mg.Terminal))
        self.assertFalse(isinstance(t, mg.Nonterminal))

        self.assertTrue(isinstance(nt, mg.Symbol))
        self.assertTrue(isinstance(nt, mg.Nonterminal))
        self.assertFalse(isinstance(nt, mg.Terminal))


class TestTrim(unittest.TestCase):
    nonterminals = [
        s := mg.START_SYMBOL,
        nt1 := mg.Nonterminal("NT1"),
        nt2 := mg.Nonterminal("NT2"),
        nt3 := mg.Nonterminal("NT3"),
        nt4 := mg.Nonterminal("NT4"),
    ]

    terminals = [
        y := mg.Terminal("y"),
        l := mg.Terminal("l"),  # noqa E741
        u := mg.Terminal("u"),
        e := mg.Terminal("e"),
        g := mg.Terminal("g"),
    ]

    productions = [
        mg.Production(s, (y,)),
        mg.Production(nt4, (nt1, nt2)),
        mg.Production(nt4, (l,)),
        mg.Production(nt4, (u,)),
        mg.Production(nt4, (e,)),
        mg.Production(nt4, (g,)),
        mg.Production(nt1, (nt3, nt2)),
        mg.Production(nt3, (nt3, nt1)),
        mg.Production(nt2, (nt3, nt1)),
    ]

    def test_compute_trim_set(self):
        trim_set_computed = mg.compute_trim_set(
            productions=self.productions,
            nonterminals=self.nonterminals,
            terminals=self.terminals,
        )
        trim_set_expected = set(self.s)
        self.assertEqual(trim_set_computed, trim_set_expected)

    def test_compute_usable_prods(self):
        trim_set = set(self.s)
        usable_prods_computed = mg.compute_usable_prods(
            trim_set=trim_set, productions=self.productions
        )
        usable_prods_expected = [mg.Production(self.s, (mg.Terminal("y"),))]
        self.assertEqual(usable_prods_computed, usable_prods_expected)


def main():
    unittest.main()


if __name__ == "__main__":
    main()
