"""Context-free grammar.

@TODO: Add support for generating regular expressions.
@TODO: Make random seed configurable.
@TODO: Add proper docstrings.
"""

import pathlib
import random
from enum import Enum
from typing import Iterator, List, Tuple, Union

import exrex
import nltk
import numpy as np
import pyrootutils

Nonterminal = nltk.Nonterminal
Symbol = Union[str, Nonterminal]
ProbabalisticProduction = nltk.ProbabilisticProduction

PROJECT_ROOT = path = pyrootutils.find_root(
    search_from=__file__, indicator=".project-root"
)


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
    def sample_cfg(
        cls,
        n_terminals: int,
        n_nonterminals: int,
        data_dir: str = PROJECT_ROOT / "data",
        filename: str = "sample",
        lp: float = 0.5,
        bp: float = 0.5,
    ):
        """Samples a random CFG and saves it to a file.

        See implementation at https://github.com/alexc17/syntheticpcfg/blob/master/syntheticpcfg/sample_grammar.py

        @TODO: This will usually generate a bad CFG, in the sense that a lot of
        productions are useless. Also, there's not real distributional control yet.
        """

        terminals = [f"t{i}" for i in range(n_terminals)]
        nonterminals = ["S"] + [f"NT{i}" for i in range(n_nonterminals)]
        productions = []
        for a in nonterminals:
            for b in terminals:
                if np.random.random() < lp:
                    productions.append(f"{a} -> '{b}'")
        for a in nonterminals:
            for b in nonterminals[1:]:
                for c in nonterminals[1:]:
                    if np.random.random() < bp:
                        productions.append(f"{a} -> {b} {c}")

        with open(data_dir / f"{filename}.cfg", "w") as f:
            f.write("\n".join(productions))

    @classmethod
    def from_grammar(cls, grammar_file: pathlib.Path | str):
        grammar = cls()

        if isinstance(grammar_file, str):
            grammar_file = pathlib.Path(grammar_file)

        # Load grammar and set type
        ext: str = grammar_file.suffix
        if (ext == ".cfg") or (ext == ".pcfg"):
            grammar.type = cls.Type.CFG
        elif ext == ".regex":
            grammar.type = cls.Type.Regular
        else:
            raise ValueError(f"Unknown grammar type: {ext}")

        with open(grammar_file, "r") as f:
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
