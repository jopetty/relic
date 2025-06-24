import unittest

from formal_gym.metaxbargrammar import GrammarParams, XBarGrammar


class TestXbar(unittest.TestCase):
    def test_create_grammar_params(self):
        grammar_params: GrammarParams = GrammarParams(
            head_initial=True,
            spec_initial=True,
            pro_drop=False,
            proper_with_det=True,
            avg_syllables=2,
            max_consonants=3,
            verbs=3,
            nouns=3,
            propns=3,
            prons=4,
            adjs=4,
            det_def=1,
            det_indef=1,
            comps=1,
        )
        self.assertEqual(grammar_params.head_initial, True)
        self.assertEqual(grammar_params.spec_initial, True)
        self.assertEqual(grammar_params.pro_drop, False)
        self.assertEqual(grammar_params.proper_with_det, True)
        self.assertEqual(grammar_params.avg_syllables, 2)
        self.assertEqual(grammar_params.max_consonants, 3)
        self.assertEqual(grammar_params.verbs, 3)

    def test_create_grammar(self):
        grammar_params: GrammarParams = GrammarParams(
            head_initial=True,
            spec_initial=True,
            pro_drop=False,
            proper_with_det=True,
            avg_syllables=2,
            max_consonants=3,
        )
        grammar: XBarGrammar = XBarGrammar.from_params(grammar_params)
