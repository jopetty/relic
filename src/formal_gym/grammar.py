"""Context-free grammar.

@TODO: Add support for generating regular expressions.
@TODO: Make random seed configurable.
@TODO: Add proper docstrings.
"""

import pathlib
import random
from enum import Enum
from typing import Iterator, List, Optional, Tuple

import exrex
import nltk
import pyrootutils

Terminal = str
Nonterminal = nltk.Nonterminal
Symbol = Terminal | Nonterminal
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

    @property
    def terminals(self) -> List[str]:
        lexical_rules = [t for t in self.grammar_obj.productions() if t.is_lexical()]
        terminals = [r.rhs()[0] for r in lexical_rules]
        return terminals

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

    def generate_negative_sample(
        self, s_max_length: int = 10, random_length: bool = True
    ) -> Optional[str]:
        if self.type != self.Type.CFG:
            raise ValueError("Negative examples are only supported for CFGs.")

        parser = nltk.parse.ChartParser(self.grammar_obj)

        if random_length:
            str_len = random.randint(1, s_max_length)
        else:
            str_len = s_max_length
        is_parsable = True
        max_trials = 20
        while is_parsable and max_trials > 0:
            max_trials -= 1
            # print(max_trials)
            ex = [random.choice(self.terminals) for _ in range(str_len)]
            parses = list(parser.parse(ex))
            if len(parses) == 0:
                is_parsable = False

        if not is_parsable:
            return " ".join(ex)
        else:
            return None

    def generate_negative_samples(
        self, n_samples: int, s_max_length: int = 10
    ) -> List[str]:
        uniuqe_samples = set()
        while len(uniuqe_samples) < n_samples:
            uniuqe_samples.add(self.generate_negative_sample(s_max_length=s_max_length))

        return list(uniuqe_samples)

    def generate_negative_samples_matching_lengths(
        self,
        positive_sample_path: pathlib.Path | str,
    ):
        if isinstance(positive_sample_path, str):
            positive_sample_path = pathlib.Path(positive_sample_path)

        with open(positive_sample_path, "r") as f:
            positive_samples = f.readlines()

        lengths = [len(s.split(" ")) for s in positive_samples]
        unique_negative_samples = set()

        for length in lengths:
            max_trials = 20
            neg_sample = self.generate_negative_sample(
                s_max_length=length, random_length=False
            )
            if neg_sample is None:
                continue
            while neg_sample in unique_negative_samples and max_trials > 0:
                max_trials -= 1
                neg_sample = self.generate_negative_sample(
                    s_max_length=length, random_length=False
                )
            unique_negative_samples.add(neg_sample)

        return list(unique_negative_samples)

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
