"""Context-free grammar."""

import pathlib
import random
from enum import Enum
from typing import Iterator, List, Tuple, Union

import exrex
import nltk

Nonterminal = nltk.Nonterminal
Symbol = Union[str, Nonterminal]
ProbabalisticProduction = nltk.ProbabilisticProduction


class Grammar:
    """Wrapper for various kinds of generative grammars.

    Implementation for the `generate()` method is dependent on the type of the grammar,
    which is inferred at creation from the suffix of the grammar file.

    CFG generation is based on Thomas Breydo's PCFG package `pcfg`: https://github.com/thomasbreydo/pcfg/blob/master/pcfg/pcfg.py
    """

    class Type(Enum):
        CFG = "cfg"
        Regular = "regular"

    @classmethod
    def from_grammar(cls, grammar: pathlib.Path):
        grammar = cls()

        # Load grammar and set type
        ext: str = grammar.suffix
        if (ext == ".cfg") or (ext == ".pcfg"):
            grammar.type = cls.Type.CFG
        elif ext == ".regex":
            grammar.type = cls.Type.Regular
        else:
            raise ValueError(f"Unknown grammar type: {ext}")

        with open(grammar, "r") as f:
            if ext == ".cfg":
                grammar.grammar_obj = nltk.CFG.fromstring(f.read())
            elif ext == ".pcfg":
                grammar.grammar_obj = nltk.PCFG.fromstring(f.read())
            elif ext == ".regex":
                grammar.grammar_obj = f.read()
            else:
                raise ValueError(f"Unknown grammar type: {ext}")

        return grammar

    def generate(self, n_samples: int, sep: str = "") -> Iterator[str]:
        if self.type == self.Type.CFG:
            for _ in range(n_samples):
                yield self._generate_derivation(self.grammar_obj.start(), sep=sep)
        elif self.type == self.Type.Regular:
            for _ in range(n_samples):
                yield exrex.getone(self.grammar_obj)

    def _generate_derivation(self, nonterminal: Nonterminal, sep: str) -> str:
        sentence: List[str] = []
        symbol: Symbol
        derivation: str
        for symbol in self._reduce_once(nonterminal):
            if isinstance(symbol, str):
                derivation = symbol
            else:
                derivation = self._generate_derivation(symbol, sep=sep)
            if derivation != "":
                sentence.append(derivation)
        return sep.join(sentence)

    def _reduce_once(self, nonterminal: Nonterminal) -> Tuple[Symbol]:
        return self._choose_production_reducing(nonterminal).rhs()

    def _choose_production_reducing(
        self, nonterminal: Nonterminal
    ) -> ProbabalisticProduction:
        productions: List[ProbabalisticProduction] = self.grammar_obj._lhs_index[
            nonterminal
        ]
        probabilities: List[float]

        if isinstance(self.grammar_obj, nltk.PCFG):
            probabilities = [p.prob() for p in productions]
        else:
            p = 1.0 / len(productions)
            probabilities = [p for _ in productions]

        return random.choices(productions, weights=probabilities)[0]
