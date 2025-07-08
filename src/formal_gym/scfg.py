import random
import re
from typing import Dict, List, Optional, Set, Tuple

import formal_gym.metaxbargrammar as fg_mxg

SyncGrammarParams = fg_mxg.SyncGrammarParams
Rule = Tuple[Tuple[str, ...], Tuple[str, ...]]


class SCFG:
    """
    A class to represent and sample from a Synchronous Context-Free Grammar.
    It parses grammar rules and provides a method to generate synchronized
    string pairs, with control over recursion depth.
    """

    def __init__(
        self, sync_params: SyncGrammarParams | None = None, sync_str: str | None = None
    ):
        """
        Initializes the SCFG by parsing the grammar rules from SyncGrammarParams.

        Args:
            sync_params: An object containing the synchronized grammar info.
            sync_str: A string containing the synchronized grammar info.
        """

        if sync_str is None and sync_params is None:
            raise ValueError("Either sync_params or sync_str must be provided")

        if sync_params is not None:
            grammar_str = sync_params.as_cfg_str()
        else:
            grammar_str = sync_str
        self.rules: Dict[str, List[Rule]] = self._parse_rules(grammar_str)
        self.start_symbol: str = "S"
        # Define symbols that increase recursion depth (e.g., clausal complements)
        self.recursive_symbols: Set[str] = {"CP_matrix", "CP_embed"}

    def _parse_rules(self, grammar_str: str) -> Dict[str, List[Rule]]:
        """
        Parses a string representation of an SCFG into a dictionary format.
        Example output: {'TBAR': [(('T', 'VP'), ('VP', 'T'))]}
        """
        rules: Dict[str, List[Rule]] = {}
        for line in grammar_str.strip().split("\n"):
            line = line.strip()
            if not line or "->" not in line:
                continue

            lhs, rhs_str = map(str.strip, line.split("->", 1))

            # Extract left and right productions from the < > bracketed part.
            match = re.search(r"<(.*),\s*(.*)>", rhs_str)
            if not match:
                continue
            left_rhs_str, right_rhs_str = match.groups()

            # Split into symbols (non-terminals or 'terminals').
            left_symbols = tuple(left_rhs_str.strip().split())
            right_symbols = tuple(right_rhs_str.strip().split())

            if lhs not in rules:
                rules[lhs] = []
            rules[lhs].append((left_symbols, right_symbols))
        return rules

    def sample(
        self, max_depth: int = 1, rng: Optional[random.Random] = None
    ) -> Dict[str, str]:
        """
        Samples a synchronized pair of strings from the grammar.

        Args:
            max_depth: The maximum nesting depth for recursive rules.
            rng: An optional random number generator for deterministic sampling.

        Returns:
            A dictionary with full and phonetic (null-filtered) derivations, and parse trees.
        """
        if rng is None:
            rng = random.Random()

        # The recursive helper now returns six values
        left_full, left_phon, right_full, right_phon, left_tree, right_tree = (
            self._sample_recursive(
                self.start_symbol, rng, current_depth=0, max_depth=max_depth
            )
        )

        # Clean up whitespace in all generated strings before returning
        left_full = " ".join(left_full.split())
        left_phon = " ".join(left_phon.split())
        right_full = " ".join(right_full.split())
        right_phon = " ".join(right_phon.split())
        left_tree = " ".join(left_tree.split())
        right_tree = " ".join(right_tree.split())
        return {
            "left": left_full,
            "left_phonetic": left_phon,
            "right": right_full,
            "right_phonetic": right_phon,
            "left_tree": left_tree,
            "right_tree": right_tree,
        }

    def _sample_recursive(
        self, symbol: str, rng: random.Random, current_depth: int, max_depth: int
    ) -> Tuple[str, str, str, str, str, str]:
        """
        Recursively expands a symbol, returning full and phonetic derivations, and parse trees.

        Args:
            symbol: The non-terminal symbol to expand.
            rng: The random number generator.
            current_depth: The current recursion depth.
            max_depth: The maximum allowed recursion depth.

        Returns:
            A tuple of six strings: (left_full, left_phonetic, right_full, right_phonetic, left_tree, right_tree).
        """
        # Base case: The symbol is a terminal.
        if symbol not in self.rules:
            clean_symbol: str = symbol.strip("'")
            full_string: str = clean_symbol
            # Phonetic string is empty if it's a null symbol.
            phonetic_string = "" if clean_symbol.startswith("∅") else full_string
            # For a terminal, left and right derivations are identical.
            return (
                full_string,
                phonetic_string,
                full_string,
                phonetic_string,
                full_string,
                full_string,
            )

        # Recursive step: The symbol is a non-terminal.
        possible_rules: list[Rule] = self.rules[symbol]

        if current_depth >= max_depth:
            possible_rules = [
                rule
                for rule in possible_rules
                if not any(s in self.recursive_symbols for s in rule[0])
            ]
            if not possible_rules:
                return (
                    "",
                    "",
                    "",
                    "",
                    f"({symbol})",
                    f"({symbol})",
                )  # Cannot expand further

        chosen_left_prod, chosen_right_prod = rng.choice(possible_rules)

        # This dictionary will store the full (6-part) derivations for sub-trees.
        sub_derivations: Dict[str, Tuple[str, str, str, str, str, str]] = {}
        unique_non_terminals: list[str] = []
        seen: set[str] = set()
        for s in chosen_left_prod + chosen_right_prod:  # keeps lhs/rhs order
            if s in self.rules and s not in seen:
                unique_non_terminals.append(s)
                seen.add(s)

        for s in unique_non_terminals:
            new_depth = current_depth + (1 if s in self.recursive_symbols else 0)
            sub_derivations[s] = self._sample_recursive(s, rng, new_depth, max_depth)

        # Assemble the four component strings and two trees.
        left_full, left_phon = [], []
        right_full, right_phon = [], []
        left_tree_parts, right_tree_parts = [], []

        # Assemble left-side strings and tree.
        for s_left in chosen_left_prod:
            if s_left in sub_derivations:  # Non-terminal
                l_full, l_phon, _, _, l_tree, _ = sub_derivations[s_left]
                left_full.append(l_full)
                left_phon.append(l_phon)
                left_tree_parts.append(l_tree)
            else:  # Terminal
                clean_symbol = s_left.strip("'")
                left_full.append(clean_symbol)
                if not clean_symbol.startswith("∅"):
                    left_phon.append(clean_symbol)
                left_tree_parts.append(clean_symbol)

        # Assemble right-side strings and tree.
        for s_right in chosen_right_prod:
            if s_right in sub_derivations:  # Non-terminal
                _, _, r_full, r_phon, _, r_tree = sub_derivations[s_right]
                right_full.append(r_full)
                right_phon.append(r_phon)
                right_tree_parts.append(r_tree)
            else:  # Terminal
                clean_symbol = s_right.strip("'")
                right_full.append(clean_symbol)
                if not clean_symbol.startswith("∅"):
                    right_phon.append(clean_symbol)
                right_tree_parts.append(clean_symbol)

        left_tree = f"({symbol} {' '.join(left_tree_parts)})"
        right_tree = f"({symbol} {' '.join(right_tree_parts)})"

        return (
            " ".join(left_full),
            " ".join(left_phon),
            " ".join(right_full),
            " ".join(right_phon),
            left_tree,
            right_tree,
        )


if __name__ == "__main__":
    # Initialize the grammar and SCFG object.
    eng_ger_params = SyncGrammarParams.english_german()
    scfg = SCFG(eng_ger_params)

    rng = random.Random(42)

    # Sample with a fixed seed for reproducibility.
    print("--- Deterministic Sampling (max_depth=4) ---")
    sample = scfg.sample(max_depth=4, rng=rng)

    print(f"Left (Full):      {sample['left']}")
    print(f"Left (Phonetic):  {sample['left_phonetic']}")
    print(f"Right (Full):     {sample['right']}")
    print(f"Right (Phonetic): {sample['right_phonetic']}")
    print(f"Left Tree:         {sample['left_tree']}")
    print(f"Right Tree:        {sample['right_tree']}")
