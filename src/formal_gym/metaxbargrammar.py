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
    tense_lex: List[str] = field(default_factory=lambda: ["past", "pres"])
    asp_lex: List[str] = field(default_factory=lambda: ["prog", "perf"])
    agrs_lex: List[str] = field(default_factory=lambda: ["agrS0", "agrS1"])
    agro_lex: List[str] = field(default_factory=lambda: ["agrO0", "agrO1"])

    # fallback sizes
    n_verbs: int = field(init=False, default=3)
    n_nouns: int = field(init=False, default=3)
    n_propns: int = field(init=False, default=3)
    n_adjs: int = field(init=False, default=2)
    n_comps: int = field(init=False, default=2)
    n_wh: int = field(init=False, default=2)

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

        if self.comp_lex is None:
            self.comp_lex = [f"c{i}" for i in range(self.n_comps)]
        else:
            self.n_comps = len(self.comp_lex)

        if self.wh_lex is None:
            self.wh_lex = [f"wh{i}" for i in range(self.n_wh)]
        else:
            self.n_wh = len(self.wh_lex)

    # handy english preset
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
            propn_lex=["john", "mary", "london"],
            adj_lex=["big", "small", "red", "green", "blue", "fuzzy"],
            det_lex=["the", "a"],
            comp_lex=["that"],
            wh_lex=["who", "what", "where"],
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

    # wh‑root vs declarative root
    if p.wh_movement:
        # CP shell with wh spec and null C head
        rules += _shell_rules(
            "CP", "WH", "CNULL", "TP", head_initial=p.comp_initial, spec_first=True
        )
    else:
        # declarative only: CP shell with null C head
        rules += _shell_rules(
            "CP", "CNULL", "CNULL", "TP", head_initial=p.comp_initial, spec_first=True
        )

    # embedded CP with overt C (binary)
    rules.append("CP -> C TP" if p.comp_initial else "CP -> TP C")

    # ----- TP shell -----
    rules += _shell_rules(
        "TP", "NP_SUBJ", "T_HEAD", "VP", head_initial=p.verb_raise, spec_first=True
    )
    # T_HEAD expands to T (and optional AGR_S if rich)
    if p.rich_agreement:
        rules += [
            "T_HEAD -> T AGR_S" if p.verb_raise else "T_HEAD -> AGR_S T",
        ]
    else:
        rules.append("T_HEAD -> T")

    # subject licensing
    rules.append("NP_SUBJ -> PRO | NP" if p.pro_drop else "NP_SUBJ -> NP")

    # ----- VP shell (AspP spec) -----
    rules += _shell_rules(
        "VP",
        "ASP",
        "V_HEAD",
        "OBJ_PHRASE",
        head_initial=p.head_initial,
        spec_first=True,
    )

    # V_HEAD expands depending on agreement
    if p.rich_agreement:
        rules.append("V_HEAD -> V AGR_O")
    else:
        rules.append("V_HEAD -> V")

    # OBJ_PHRASE may shift out (object_shift parameter)
    if p.object_shift:
        # object raises: little_vP shell handles it; here OBJ_PHRASE is empty
        rules.append("OBJ_PHRASE -> EPS_OBJ")
        rules.append("EPS_OBJ -> '∅'")
    else:
        # object stays
        rules.append("OBJ_PHRASE -> NP")

    # ----- Nominal shells (NP) -----
    # determiner + N shell
    if p.proper_with_det:
        # allow DET PROPN variant
        rules += _shell_rules(
            "NP", "DET", "N_HEAD", "ADJLIST", head_initial=False, spec_first=True
        )
        rules.append("N_HEAD -> PROPN")
    else:
        rules += _shell_rules(
            "NP", "DET", "N_HEAD", "ADJLIST", head_initial=False, spec_first=True
        )
        rules.append("N_HEAD -> N | PROPN")

    # adjectives recurse
    rules += [
        "ADJLIST -> ADJ",
        "ADJLIST -> ADJ ADJLIST",
    ]

    # ----- “little vP” shell for object shift (optional) -----
    if p.object_shift:
        rules += _shell_rules(
            "little_vP",
            "NP",
            "V_HEAD",
            "EPS_OBJ",
            head_initial=p.head_initial,
            spec_first=True,
        )

    # ----- Lexicon -----
    rules += _lex("DET", p.det_lex if p.det_lex else [f"det{i}" for i in range(2)])
    rules += _lex("T", [f"t_{x}" for x in p.tense_lex])
    rules += _lex("ASP", [f"asp_{x}" for x in p.asp_lex])
    rules += _lex("C", p.comp_lex)
    rules += _lex("CNULL", ["∅C"])
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

    grammar_str = "\n".join(rules)

    # DO NOT ATTEMPT TO DO ANYTHING WITH fg_grammar HERE
    # SIMPLY RETURN THE STRING, THAT IS ALL
    # **DO NOT EDIT THE COMMENTS**
    return grammar_str  # DO NOT EDIT THIS LINE


# BEGIN FG BLOCK: CUSTOM IMPORTS. **DO NOT EDIT**
# **DO NOT EVEN EDIT THE COMMENTS**
if __name__ == "__main__":  # DO NOT EDIT THIS LINE
    english_params = GrammarParams.english()  # DO NOT EDIT THIS LINE
    print("Running with params:")  # DO NOT EDIT THIS LINE
    pprint.pprint(asdict(english_params))  # DO NOT EDIT THIS LINE
    english_grammar_str = generate_cfg(english_params)  # DO NOT EDIT THIS LINE
    # print(english_grammar_str)  # DO NOT EDIT THIS LINE
    english_grammar = fg_grammar.Grammar.from_string(  # DO NOT EDIT THIS LINE
        english_grammar_str,  # DO NOT EDIT THIS LINE
        grammar_type=GType.CFG,  # DO NOT EDIT THIS LINE
    )  # DO NOT EDIT THIS LINE

    print(english_grammar.as_cfg)  # DO NOT EDIT THIS LINE

    for _ in range(2):  # DO NOT EDIT THIS LINE
        sample = english_grammar.generate_tree()  # DO NOT EDIT THIS LINE
        print(sample["string"])  # DO NOT EDIT THIS LINE
        print(sample["parse"])  # DO NOT EDIT THIS LINE
# END FG BLOCK
