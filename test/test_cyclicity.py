import unittest

from formal_gym import grammar as fg_grammar


class TestSymbols(unittest.TestCase):
    CYCLIC_1 = """S -> A B
    A -> A C
    B -> 't'
    C -> 'c'
    """
    CYCLIC_2 = """S -> A B
    A -> B C
    B -> A D
    A -> 'a'
    B -> 'b'
    C -> 'c'
    D -> 'd'
    """
    CYCLIC_3 = """
    S -> A B
    A -> C D
    D -> B F
    B -> A E
    C -> 'c'
    E -> 'e'
    F -> 'f'"""

    ACYCLIC_1 = """S -> A B
    A -> 'a'
    B -> 'b'
    """

    def test_cyclic_grammars(self):
        grammar_1 = fg_grammar.Grammar.from_string(
            self.CYCLIC_1, grammar_type=fg_grammar.Grammar.Type.CFG
        )
        self.assertTrue(grammar_1.is_cyclic)

        grammar_2 = fg_grammar.Grammar.from_string(
            self.CYCLIC_2, grammar_type=fg_grammar.Grammar.Type.CFG
        )
        self.assertTrue(grammar_2.is_cyclic)

        grammar_3 = fg_grammar.Grammar.from_string(
            self.CYCLIC_3, grammar_type=fg_grammar.Grammar.Type.CFG
        )
        self.assertTrue(grammar_3.is_cyclic)

    def test_acyclic_grammars(self):
        grammar = fg_grammar.Grammar.from_string(
            self.ACYCLIC_1, grammar_type=fg_grammar.Grammar.Type.CFG
        )
        self.assertFalse(grammar.is_cyclic)


def main():
    unittest.main()


if __name__ == "__main__":
    main()
