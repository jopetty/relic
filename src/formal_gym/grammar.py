"""Context-free grammar.

@TODO: Add support for generating regular expressions.
@TODO: Make random seed configurable.
@TODO: Add proper docstrings.
"""

import pathlib
import random
from collections import defaultdict
from enum import Enum
from typing import Iterator, List, Optional, Set, Tuple

import exrex
import nltk
import pyrootutils

Terminal = str
Nonterminal = nltk.Nonterminal
Symbol = Terminal | Nonterminal
Production = nltk.Production
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

    def test_sample(self, sample: str) -> bool:
        if self.type != self.Type.CFG:
            raise ValueError("Testing samples is only supported for CFGs.")

        parser = nltk.ChartParser(self.grammar_obj)
        parses = list(parser.parse(sample.split(" ")))
        return len(parses) > 0

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


class ContextFreeGrammar(Grammar):
    """Context-free-grammar."""

    @property
    def is_pcfg(self) -> bool:
        return self._is_pcfg

    @property
    def as_cfg(self) -> nltk.CFG:
        return self._nltk_cfg

    @property
    def as_pcfg(self) -> nltk.PCFG:
        return self._nltk_pcfg

    @property
    def terminals(self) -> Set[str]:
        lexical_rules = [t for t in self.as_cfg.productions() if t.is_lexical()]
        terminals = set([r.rhs()[0] for r in lexical_rules])
        return terminals

    @classmethod
    def from_file(cls, grammar_file: pathlib.Path | str) -> "ContextFreeGrammar":
        if isinstance(grammar_file, str):
            grammar_file = pathlib.Path(grammar_file)

        with open(grammar_file, "r") as f:
            grammar_str = f.read()

        if grammar_file.suffix == ".cfg":
            return ContextFreeGrammar(grammar_str, is_pcfg=False)
        elif grammar_file.suffix == ".pcfg":
            return ContextFreeGrammar(grammar_str, is_pcfg=True)
        else:
            raise ValueError("You must provide a CFG (*.cfg) or PCFG (*.pcfg) file.")

    def __init__(self, grammar_str: str, is_pcfg: bool = False):
        self._is_pcfg = is_pcfg

        if is_pcfg:
            self._nltk_pcfg = nltk.PCFG.fromstring(grammar_str)

            # Convert PCFG to CFG by stripping out the probabilities from each
            # production rule
            cfg_productions = [
                nltk.Production(p.lhs(), p.rhs()) for p in self._nltk_pcfg.productions()
            ]
            self._nltk_cfg = nltk.CFG(self._nltk_pcfg.start(), cfg_productions)
        else:
            self._nltk_cfg = nltk.CFG.fromstring(grammar_str)

            # Convert CFG to PCFG by assigning each rule probability 1/n, where
            # n is the number of RHS rules for the LHS nonterminal
            cfg_rules = {}
            for p in self._nltk_cfg.productions():
                cfg_rules.setdefault(p.lhs(), []).append(p.rhs())

            pcfg_rules = [
                (lhs, rhs, 1 / len(cfg_rules[lhs]))
                for lhs, rhs_list in cfg_rules.items()
                for rhs in rhs_list
            ]
            pcfg_productions = [
                nltk.grammar.ProbabilisticProduction(lhs=lhs, rhs=rhs, prob=prob)
                for lhs, rhs, prob in pcfg_rules
            ]
            self._nltk_pcfg = nltk.PCFG(self._nltk_cfg.start(), pcfg_productions)

        self.productions_by_lhs = defaultdict(list)
        self.probs_by_lhs = defaultdict(list)

        for prod in self.as_pcfg.productions():
            self.productions_by_lhs[prod.lhs()].append(prod)
            self.probs_by_lhs[prod.lhs()].append(prod.prob())

        self.can_terminate = self._find_terminating_nts()

    def _find_terminating_nts(self) -> Set[Nonterminal]:
        can_terminate = set()

        # First, add all non-terminals that only derive terminals
        for prod in self.as_cfg.productions():
            if prod.is_lexical():
                can_terminate.add(prod.lhs())

        changed = True
        while changed:
            changed = False
            for prod in self.as_cfg.productions():
                if prod.lhs() not in can_terminate:
                    if all(
                        not isinstance(sym, Nonterminal) or sym in can_terminate
                        for sym in prod.rhs()
                    ):
                        can_terminate.add(prod.lhs())
                        changed = True
        return can_terminate

    def _choose_production(
        self, lhs: Nonterminal, depth: int, max_depth: int
    ) -> Optional[Production]:
        productions = self.productions_by_lhs[lhs]
        probs = self.probs_by_lhs[lhs]

        if depth >= max_depth:
            # Only consider terminating productions
            valid_prods = [
                (p, prob)
                for p, prob in zip(productions, probs)
                if all(
                    not isinstance(sym, Nonterminal) or sym in self.can_terminate
                    for sym in p.rhs()
                )
            ]

            if not valid_prods:
                return None

            total_prob = sum(prob for _, prob in valid_prods)
            valid_prods = [(p, prob / total_prob) for p, prob in valid_prods]
            return random.choices(
                [p for p, _ in valid_prods], weights=[prob for _, prob in valid_prods]
            )[0]

        return random.choices(productions, weights=probs)[0]

    def generate(self, sep: str = " ", max_depth: int = 50) -> Optional[str]:
        def _sample_recursive(symbol: Nonterminal, depth: int) -> Optional[List[str]]:
            if depth > max_depth:
                return None

            if not isinstance(symbol, Nonterminal):
                return [str(symbol)]

            production = self._choose_production(symbol, depth, max_depth)
            if production is None:
                return None

            result = []
            for sym in production.rhs():
                if isinstance(sym, Nonterminal):
                    subsample = _sample_recursive(sym, depth + 1)
                    if subsample is None:
                        return None
                    result.extend(subsample)
                else:
                    result.append(str(sym))
            return result

        result = _sample_recursive(self.as_cfg.start(), 0)
        if result is None:
            return self.generate(sep=sep, max_depth=max_depth)
        else:
            return sep.join(result)

    def test_sample(self, sample: str) -> bool:
        parser = nltk.ChartParser(self.as_cfg)
        parses = list(parser.parse(sample.split(" ")))
        return len(parses) > 0
