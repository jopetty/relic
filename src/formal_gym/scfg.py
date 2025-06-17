# scfg.py
#
# Provides an interface to sample strings from a synchronous context-free grammar.

import random
from typing import Dict, List, Tuple

from formal_gym.metaxbargrammar import SyncGrammarParams, generate_scfg


class SCFG:
    """Stochastic sampler for a synchronous CFG."""

    def __init__(self, sync_params: SyncGrammarParams, *, max_depth: int = 50):
        # build one single SCFG, where each production is a <left-,right-> pair
        scfg_str = generate_scfg(sync_params)

        self.rules = self._parse_scfg(scfg_str)
        self.max_depth = max_depth

    def _parse_scfg(self, txt: str) -> Dict[str, List[Tuple[List[str], List[str]]]]:
        rules: Dict[str, List[Tuple[List[str], List[str]]]] = {}
        for line in txt.splitlines():
            line = line.strip()
            if not line:
                continue
            lhs, rhs = line.split("->", 1)
            lhs = lhs.strip()
            rhs = rhs.strip()
            # rhs looks like  <A B, C D>    or   <A, C>   etc.
            assert rhs.startswith("<") and rhs.endswith(">")
            inside = rhs[1:-1]
            left_part, right_part = inside.split(",", 1)
            L = left_part.strip().split()
            R = right_part.strip().split()
            rules.setdefault(lhs, []).append((L, R))
        return rules

    def generate(self) -> Dict[str, str]:
        """Sample one synchronized pair from S."""
        Ls, Rs = self._gen("S", 0)
        return {"left": Ls.strip(), "right": Rs.strip()}

    def _gen(self, sym: str, depth: int) -> Tuple[str, str]:
        prods = self.rules.get(sym)
        if not prods:
            # terminal or unknown nonterminal
            if sym.startswith("'") and sym.endswith("'"):
                tok = sym[1:-1]
                return tok, tok
            return sym, sym

        # if we've recursed too deeply, drop any production
        # that has a recursive call on 'sym' in its RHS
        if depth >= self.max_depth:
            nonrec = [(L, R) for (L, R) in prods if sym not in L and sym not in R]
            if nonrec:
                prods = nonrec

        idx = random.randrange(len(prods))
        left_rhs, right_rhs = prods[idx]

        left_pieces: List[str] = []
        right_pieces: List[str] = []
        for ltok, rtok in zip(left_rhs, right_rhs):
            # expand left side
            if ltok.startswith("'") and ltok.endswith("'"):
                left_pieces.append(ltok[1:-1])
            else:
                Ls, _ = self._gen(ltok, depth + 1)
                left_pieces.append(Ls)

            # expand right side
            if rtok.startswith("'") and rtok.endswith("'"):
                right_pieces.append(rtok[1:-1])
            else:
                _, Rs = self._gen(rtok, depth + 1)
                right_pieces.append(Rs)

        return " ".join(left_pieces), " ".join(right_pieces)


if __name__ == "__main__":
    sync_p = SyncGrammarParams.english_german()
    scfg = SCFG(sync_p)
    pair = scfg.generate()
    print("⟨EN⟩", pair["left"])
    print("⟨DE⟩", pair["right"])
