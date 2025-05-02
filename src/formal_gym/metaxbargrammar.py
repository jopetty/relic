# metaxbargrammar.py

import pprint
from dataclasses import asdict, dataclass, field
from typing import List, Optional

# BEGIN FG BLOCK: CUSTOM IMPORTS. **DO NOT EDIT**
# **DO NOT EVEN EDIT THE COMMENTS**
import formal_gym.grammar as fg_grammar

GType = fg_grammar.Grammar.Type
# END FG BLOCK

# ------------------------------------------------------------------
#  HELPER: BUILD LARSON‑STYLE SHELL RULES (CNF)
# ------------------------------------------------------------------


def _shell_rules(
    phrase: str,
    spec: str,
    head: str,
    comp: str,
    *,
    head_initial: bool = True,
    spec_first: bool = True,
):
    """Return the two CNF rules for a Larson shell.

    XP -> SPEC XBAR   /  XBAR SPEC
    XBAR -> X⁰ COMP   /  COMP X⁰
    """
    xbar = f"{head.upper()}BAR"  # unique bar label
    rules = []

    # specifier placement
    if spec_first:
        rules.append(f"{phrase} -> {spec} {xbar}")
    else:
        rules.append(f"{phrase} -> {xbar} {spec}")

    # head direction
    if head_initial:
        rules.append(f"{xbar} -> {head} {comp}")
    else:
        rules.append(f"{xbar} -> {comp} {head}")

    return rules


# ------------------------------------------------------------------
#  PARAMETER BUNDLE
# ------------------------------------------------------------------
@dataclass
class GrammarParams:
    # syntactic switches
    head_initial: bool = True
    spec_first: bool = True
    comp_initial: bool = True
    wh_movement: bool = True
    pro_drop: bool = False
    verb_raise: bool = False
    object_shift: bool = False
    rich_agreement: bool = False
    proper_with_det: bool = False

    # lexical items
    verb_lex: Optional[List[str]] = None
    noun_lex: Optional[List[str]] = None
    propn_lex: Optional[List[str]] = None
    adj_lex: Optional[List[str]] = None
    det_lex: Optional[List[str]] = None
    comp_lex: Optional[List[str]] = None
    wh_lex: Optional[List[str]] = None
    tense_lex: List[str] = field(default_factory=lambda: ["∅_T_past", "∅_T_pres"])
    asp_lex: List[str] = field(default_factory=lambda: ["∅_Asp_prog", "∅_Asp_perf"])
    agrs_lex: List[str] = field(
        default_factory=lambda: ["∅_Agrs_agrS0", "∅_Agrs_agrS1"]
    )
    agro_lex: List[str] = field(
        default_factory=lambda: ["∅_AgrO_agrO0", "∅_AgrO_agrO1"]
    )

    # fallback sizes
    n_verbs: int = 3
    n_nouns: int = 3
    n_propns: int = 3
    n_adjs: int = 2
    n_dets: int = 2
    n_comps: int = 2
    n_wh: int = 2

    def __post_init__(self):
        # auto‑fill lexicons if missing
        if self.verb_lex is None:
            self.verb_lex = [f"verb{i}" for i in range(self.n_verbs)]
        else:
            self.n_verbs = len(self.verb_lex)

        if self.noun_lex is None:
            self.noun_lex = [f"noun{i}" for i in range(self.n_nouns)]
        else:
            self.n_nouns = len(self.noun_lex)

        if self.propn_lex is None:
            self.propn_lex = [f"name{i}" for i in range(self.n_propns)]
        else:
            self.n_propns = len(self.propn_lex)

        if self.adj_lex is None:
            self.adj_lex = [f"adj{i}" for i in range(self.n_adjs)]
        else:
            self.n_adjs = len(self.adj_lex)

        if self.det_lex is None:
            self.det_lex = [f"det{i}" for i in range(self.n_dets)]
        else:
            self.n_dets = len(self.det_lex)

        if self.comp_lex is None:
            self.comp_lex = [f"c{i}" for i in range(self.n_comps)]
        else:
            self.n_comps = len(self.comp_lex)

        if self.wh_lex is None:
            self.wh_lex = [f"wh{i}" for i in range(self.n_wh)]
        else:
            self.n_wh = len(self.wh_lex)

    # english preset
    @classmethod
    def english(cls):
        return cls(
            head_initial=True,
            spec_first=True,
            comp_initial=True,
            wh_movement=True,
            pro_drop=False,
            verb_raise=False,
            object_shift=False,
            rich_agreement=False,
            proper_with_det=False,
            verb_lex=["eat", "see", "love", "give"],
            noun_lex=["tree", "horse", "dog", "cat", "apple"],
            propn_lex=["john", "mary", "sue", "bob"],
            adj_lex=["big", "small", "red", "green", "blue", "fuzzy", "round"],
            det_lex=["the", "a"],
            comp_lex=["that"],
            wh_lex=["who", "what", "where", "when", "why"],
        )


# ------------------------------------------------------------------
#  LEXICAL RULE GENERATOR
# ------------------------------------------------------------------


def _lex(pos: str, words: List[str]) -> List[str]:
    return [f"{pos} -> '{w}'" for w in words]


# ------------------------------------------------------------------
#  CFG GENERATOR
# ------------------------------------------------------------------


def generate_cfg(p: GrammarParams) -> str:
    rules: List[str] = []

    # ----- CP / S layer -----
    rules.append("S -> CP")

    # WH‑root question rule
    if p.wh_movement:
        rules += _shell_rules(
            "CP", "WH", "CNULL", "TP", head_initial=p.comp_initial, spec_first=True
        )

    # Declarative CP with overt C
    rules.append("CP -> C TP" if p.comp_initial else "CP -> TP C")

    # Declarative CP with null C (new rule)
    rules.append("CP -> CNULL TP" if p.comp_initial else "CP -> TP CNULL")

    # ----- TP shell -----
    rules += _shell_rules(
        "TP", "NP_SUBJ", "T_HEAD", "VP", head_initial=p.verb_raise, spec_first=True
    )
    if p.rich_agreement:
        rules.append("T_HEAD -> T AGR_S" if p.verb_raise else "T_HEAD -> AGR_S T")
    else:
        rules.append("T_HEAD -> T")

    rules.append("NP_SUBJ -> PRO | NP" if p.pro_drop else "NP_SUBJ -> NP")

    # ----- VP shell -----
    rules += _shell_rules(
        "VP",
        "ASP",
        "V_HEAD",
        "OBJ_PHRASE",
        head_initial=p.head_initial,
        spec_first=True,
    )

    if p.rich_agreement:
        rules.append("V_HEAD -> V AGR_O")
    else:
        rules.append("V_HEAD -> V")

    if p.object_shift:
        rules.append("OBJ_PHRASE -> EPS_OBJ")
        rules.append("EPS_OBJ -> '∅'")
    else:
        rules.append("OBJ_PHRASE -> NP")

    # ----- NP shell -----
    rules += _shell_rules(
        "NP", "DET", "N_HEAD", "ADJLIST", head_initial=False, spec_first=True
    )
    if p.proper_with_det:
        rules.append("N_HEAD -> PROPN")
    else:
        rules.append("N_HEAD -> N | PROPN")

    rules += [
        "ADJLIST -> ADJ",
        "ADJLIST -> ADJ ADJLIST",
    ]

    # ----- Lexicon -----
    rules += _lex("DET", p.det_lex)
    rules += _lex("T", [f"t_{x}" for x in p.tense_lex])
    rules += _lex("ASP", [f"asp_{x}" for x in p.asp_lex])
    rules += _lex("C", p.comp_lex)
    rules += _lex("CNULL", ["∅_C"])
    rules += _lex("WH", p.wh_lex)
    rules += _lex("V", p.verb_lex)
    rules += _lex("N", p.noun_lex)
    rules += _lex("PROPN", p.propn_lex)
    rules += _lex("ADJ", p.adj_lex if p.adj_lex else ["dummy_adj"])

    if p.rich_agreement:
        rules += _lex("AGR_S", p.agrs_lex)
        rules += _lex("AGR_O", p.agro_lex)
    if p.pro_drop:
        rules += _lex("PRO", ["pro"])

    return "\n".join(rules)


# BEGIN FG BLOCK: CUSTOM IMPORTS. **DO NOT EDIT**
# **DO NOT EVEN EDIT THE COMMENTS**
if __name__ == "__main__":  # DO NOT EDIT THIS LINE
    english_params = GrammarParams.english()  # DO NOT EDIT THIS LINE
    print("Running with params:")  # DO NOT EDIT THIS LINE
    pprint.pprint(asdict(english_params))  # DO NOT EDIT THIS LINE
    english_grammar_str = generate_cfg(english_params)  # DO NOT EDIT THIS LINE
    english_grammar = fg_grammar.Grammar.from_string(  # DO NOT EDIT THIS LINE
        english_grammar_str,  # DO NOT EDIT THIS LINE
        grammar_type=GType.CFG,  # DO NOT EDIT THIS LINE
    )  # DO NOT EDIT THIS LINE
    print(english_grammar.as_cfg)  # DO NOT EDIT THIS LINE
    for _ in range(2):  # DO NOT EDIT THIS LINE
        s = english_grammar.generate_tree()  # DO NOT EDIT THIS LINE
        print(s["string"])  # DO NOT EDIT THIS LINE
# END FG BLOCK
