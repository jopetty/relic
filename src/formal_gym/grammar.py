"""Context-free grammar.

@TODO: Add support for generating regular expressions.
@TODO: Make random seed configurable.
"""

import pathlib
import random
from collections import defaultdict
from enum import Enum
from typing import List, Optional, Set

import lark
import nltk
import pyrootutils
from lark.exceptions import LarkError

Terminal = str
Nonterminal = nltk.Nonterminal
Symbol = Terminal | Nonterminal
Production = nltk.Production
ProbabalisticProduction = nltk.ProbabilisticProduction

PROJECT_ROOT = path = pyrootutils.find_root(
    search_from=__file__, indicator=".project-root"
)


class Grammar:
    """Context-free-grammar."""

    class Type(Enum):
        CFG = "cfg"
        Regular = "regular"

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
    def as_lark_ebnf(self) -> str:
        """Returns the grammar in Lark EBNF format."""
        rules = defaultdict(set)

        for prod in self.as_cfg.productions():
            rules[prod.lhs()].add(prod.rhs())

        lark_rules = []
        for lhs, rhs_set in rules.items():
            rhs_rules = []
            for rhs in rhs_set:
                rhs_syms = []
                for s in rhs:
                    if isinstance(s, str):
                        rhs_syms.append(f'"{s}"')
                    elif isinstance(s, nltk.grammar.Nonterminal):
                        rhs_syms.append(f"{str(s).lower()}")
                rhs_string = " ".join(rhs_syms)
                rhs_rules.append(rhs_string)
            lark_rhs = " | ".join(s for s in list(rhs_rules))
            if str(lhs) == "S":
                lhs = "start"
            lark_rules.append(f"{str(lhs).lower()} : {lark_rhs}")

        g_suff = "%import common.WS_INLINE\n%ignore WS_INLINE"
        return "\n".join(lark_rules) + "\n" + g_suff

    @property
    def terminals(self) -> Set[str]:
        lexical_rules = [t for t in self.as_cfg.productions() if t.is_lexical()]
        terminals = set([r.rhs()[0] for r in lexical_rules])
        return terminals

    @property
    def nonterminals(self) -> Set[str]:
        non_lexical_rules = [t for t in self.as_cfg.productions() if not t.is_lexical()]
        lhs_list = [r.lhs() for r in non_lexical_rules]
        rhsa_list = [r.rhs()[0] for r in non_lexical_rules]
        rhsb_list = [r.rhs()[1] for r in non_lexical_rules]
        return set(lhs_list + rhsa_list + rhsb_list)

    @property
    def n_terminals(self) -> int:
        return len(self.terminals)

    @property
    def n_nonterminals(self) -> int:
        return len(self.nonterminals)

    @property
    def n_lexical_productions(self) -> int:
        return len([r for r in self.as_cfg.productions() if r.is_lexical()])

    @property
    def n_nonlexical_productions(self) -> int:
        return len([r for r in self.as_cfg.productions() if not r.is_lexical()])

    @property
    def parser(self) -> lark.Lark:
        return self._parser

    @property
    def is_cyclic(self) -> bool:
        """Check if the grammar contains cycles."""
        prods = {}
        for prod in self.as_cfg.productions():
            if not prod.is_lexical():
                lhs = prod.lhs()
                rhs_a, rbs_b = prod.rhs()
                prods.setdefault(str(lhs), []).extend([str(rhs_a), str(rbs_b)])

        visited = set()
        stack = set()

        def check_cycle(node):
            visited.add(node)
            stack.add(node)

            for neighbor in prods.get(node, []):
                if neighbor not in visited:
                    if check_cycle(neighbor):
                        return True
                elif neighbor in stack:
                    return True
            stack.remove(node)
            return False

        return check_cycle("S")

    @classmethod
    def from_file(cls, grammar_file: pathlib.Path | str):
        if isinstance(grammar_file, str):
            grammar_file = pathlib.Path(grammar_file)

        with open(grammar_file, "r") as f:
            grammar_str = f.read()

        if grammar_file.suffix == ".cfg":
            return cls(grammar_str, is_pcfg=False)
        elif grammar_file.suffix == ".pcfg":
            return cls(grammar_str, is_pcfg=True)
        else:
            raise ValueError("You must provide a CFG (*.cfg) or PCFG (*.pcfg) file.")

    @classmethod
    def from_string(
        cls, grammar_str: str, grammar_type: Type, with_parser: bool = True
    ):
        if grammar_type == cls.Type.CFG:
            return cls(grammar_str, is_pcfg=False, with_parser=with_parser)
        else:
            raise ValueError("Only CFGs are supported for now.")

    def __init__(
        self, grammar_str: str, is_pcfg: bool = False, with_parser: bool = True
    ):
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

        if with_parser:
            self._parser = lark.Lark(self.as_lark_ebnf)

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

    def generate(self, sep: str = " ", max_depth: int = 50) -> str:
        """Generates a single sample from the grammar.

        Args:
            sep: Separator to use between symbols.
            max_depth: Maximum depth of recursion.
        Returns:
            A dictionary with the keys "sample" and "parse" representing the
            generated sample and its parse tree.
        """

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

    def generate_tree(self, sep: str = " ", max_depth: int = 50) -> dict | None:
        """Samples a string and its parse tree.

        Args:
            sep: Separator to use between symbols.
            max_depth: Maximum depth of recursion.
        Returns:
            A dictionary with keys "string" and "parse".
        """

        def _sample_recursive(symbol: Nonterminal, depth: int):
            if depth > max_depth:
                return None

            if not isinstance(symbol, Nonterminal):
                return str(symbol), str(symbol)

            production = self._choose_production(symbol, depth, max_depth)
            if production is None:
                return None

            children_strings = []
            children_parses = []
            for sym in production.rhs():
                subsample = _sample_recursive(sym, depth + 1)
                if subsample is None:
                    return None
                children_strings.append(subsample[0])
                children_parses.append(subsample[1])

            parsed_string = sep.join(children_strings)
            parsed_tree = f"({symbol} {sep.join(children_parses)})"

            return parsed_string, parsed_tree

        try:
            result = _sample_recursive(self.as_cfg.start(), 0)
            if result is None:
                return self.generate_tree(sep=sep, max_depth=max_depth)
            else:
                return {"string": result[0], "parse": result[1]}
        except RecursionError:
            return None

    def generate_negative_sample_of_length(
        self,
        length: int,
        max_trials: int = 20,
        sep: str = " ",
    ) -> Optional[str]:
        while max_trials > 0:
            max_trials -= 1
            sample = sep.join(random.choices(list(self.terminals), k=length))
            if not self.test_sample(sample):
                return sample
        return None

    def generate_negative_sample(
        self,
        max_trials: int = 20,
        max_length: int = 50,
        sep: str = " ",
    ) -> Optional[str]:
        trials = 0
        while trials < max_trials:
            trials += 1
            str_len = random.randint(1, max_length)
            sample = sep.join(random.choices(list(self.terminals), k=str_len))
            if not self.test_sample(sample):
                return sample
        return None

    def parse(self, sample: str) -> str | None:
        def clean_parse_tree(tree, sample) -> str:
            labels = sample.split()

            def recurse(node):
                children = " ".join(recurse(child) for child in node.children)
                if children == "":
                    children = labels.pop(0)
                node_label = node.data.upper()
                if node_label == "START":
                    node_label = "S"
                return f"({node_label} {children})"

            return recurse(tree)

        try:
            parse_tree = self.parser.parse(sample)
            return clean_parse_tree(parse_tree, sample)
        except LarkError:
            return None

    def test_sample(self, sample: str) -> bool:
        """Test if a sample is valid in the grammar."""

        if self.parse(sample) is None:
            return False
        else:
            return True

    def sample_parse_depth(self, sample: str) -> int | None:
        """Count the depth of the dyck string of the sample's parse tree."""
        parse_tree = self.parse(sample)
        if parse_tree is None:
            return None
        else:
            depth = max_depth = 0
            for char in parse_tree:
                if char == "(":
                    depth += 1
                    max_depth = max(max_depth, depth)
                elif char == ")":
                    depth -= 1
            return max_depth
