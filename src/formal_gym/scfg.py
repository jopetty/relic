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

    def __init__(self, sync_params: SyncGrammarParams):
        """
        Initializes the SCFG by parsing the grammar rules from SyncGrammarParams.

        Args:
            sync_params: An object containing the synchronized grammar info.
        """
        self.rules: Dict[str, List[Rule]] = self._parse_rules(
            fg_mxg.generate_scfg(sync_params)
        )
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
            max_depth: The maximum nesting depth for recursive rules. A depth of
                       1 allows one level of embedding (e.g., a main clause with
                       an embedded clause), but not nested embeddings.
            rng: An optional random number generator for deterministic sampling.

        Returns:
            A dictionary with "left" and "right" keys holding the derived strings.
        """
        if rng is None:
            rng = random.Random()

        left_derivation, right_derivation = self._sample_recursive(
            self.start_symbol, rng, 0, max_depth
        )

        # Clean up output: remove extra whitespace and leading/trailing spaces.
        left_clean = re.sub(r"\s+", " ", left_derivation).strip()
        right_clean = re.sub(r"\s+", " ", right_derivation).strip()

        return {"left": left_clean, "right": right_clean}

    def _sample_recursive(
        self, symbol: str, rng: random.Random, current_depth: int, max_depth: int
    ) -> Tuple[str, str]:
        """
        Recursively expands a symbol to generate synchronized strings.

        This corrected version first generates all necessary child derivations
        and then assembles the final strings, correctly handling reordering.

        Args:
            symbol: The non-terminal symbol to expand.
            rng: The random number generator.
            current_depth: The current recursion depth.
            max_depth: The maximum allowed recursion depth.

        Returns:
            A tuple containing the (left_string, right_string) for the derivation.
        """
        # Base case: The symbol is a terminal (i.e., not a non-terminal).
        if symbol not in self.rules:
            clean_symbol = symbol.strip("'")
            # Return empty strings for null elements to remove them from output.
            # return ("", "") if clean_symbol.startswith('∅') else (clean_symbol, clean_symbol)
            return (clean_symbol, clean_symbol)

        # --- Recursive Step ---
        possible_rules = self.rules[symbol]

        # If we are at max depth, filter out rules that would continue recursion.
        if current_depth >= max_depth:
            possible_rules = [
                rule
                for rule in possible_rules
                if not any(s in self.recursive_symbols for s in rule[0])
            ]
            # If no non-recursive rules are available, we cannot expand further.
            if not possible_rules:
                return ("", "")

        # Choose one production rule synchronously for both sides.
        chosen_left_prod, chosen_right_prod = rng.choice(possible_rules)

        # Create synchronized derivations for all unique non-terminals in the rule.
        # This ensures that if 'NP' appears twice, we generate one 'NP' pair
        # and reuse it, but more importantly, it correctly generates pairs for
        # different non-terminals like 'T' and 'VP'.
        sub_derivations: Dict[str, Tuple[str, str]] = {}
        # We only need to find unique non-terminals to avoid redundant recursion.
        unique_non_terminals = {
            s for s in chosen_left_prod + chosen_right_prod if s in self.rules
        }

        for s in unique_non_terminals:
            new_depth = current_depth + (1 if s in self.recursive_symbols else 0)
            sub_derivations[s] = self._sample_recursive(s, rng, new_depth, max_depth)

        # Assemble the final strings by substituting the generated derivations.
        left_parts, right_parts = [], []

        for s_left in chosen_left_prod:
            # If it's a non-terminal, look up its generated left part.
            if s_left in sub_derivations:
                left_parts.append(sub_derivations[s_left][0])
            # Otherwise, it's a terminal.
            else:
                clean_symbol = s_left.strip("'")
                if not clean_symbol.startswith("∅"):
                    left_parts.append(clean_symbol)

        for s_right in chosen_right_prod:
            # Look up the corresponding right part of the derivation.
            if s_right in sub_derivations:
                right_parts.append(sub_derivations[s_right][1])
            else:
                clean_symbol = s_right.strip("'")
                if not clean_symbol.startswith("∅"):
                    right_parts.append(clean_symbol)

        return " ".join(left_parts), " ".join(right_parts)


if __name__ == "__main__":
    # 1. Initialize the grammar parameters and the SCFG object.
    eng_ger_params = SyncGrammarParams.english_german()
    scfg = SCFG(eng_ger_params)

    # 2. Sample with a fixed seed for a reproducible simple sentence.
    print("--- Deterministic Sampling (max_depth=1, seed=42) ---")
    seeded_rng = random.Random(42)
    sample1 = scfg.sample(max_depth=1, rng=seeded_rng)
    print(f"Left (EN): {sample1['left']}")
    print(f"Right (DE): {sample1['right']}\n")

    # 3. Sample a more complex sentence with recursion, also seeded.
    print("--- Deterministic Sampling (max_depth=2, seed=101) ---")
    seeded_rng_2 = random.Random(101)
    sample2 = scfg.sample(max_depth=2, rng=seeded_rng_2)
    print(f"Left (EN): {sample2['left']}")
    print(f"Right (DE): {sample2['right']}\n")

    # 4. Show a fully random sample.
    print("--- Random Sampling (max_depth=3) ---")
    sample3 = scfg.sample(max_depth=3)
    print(f"Left (EN): {sample3['left']}")
    print(f"Right (DE): {sample3['right']}")
