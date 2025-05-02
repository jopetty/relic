# metaxbargrammar.py

from dataclasses import dataclass, field
from typing import List, Optional

# BEGIN FG BLOCK: CUSTOM IMPORTS. **DO NOT EDIT**
# **DO NOT EVEN EDIT THE COMMENTS**
import formal_gym.grammar as fg_grammar

GType = fg_grammar.Grammar.Type
# END FG BLOCK


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

    # lexicon (explicit lists optional)
    verb_lex: Optional[List[str]] = None
    noun_lex: Optional[List[str]] = None
    adj_lex: Optional[List[str]] = None
    det_lex: Optional[List[str]] = None
    comp_lex: Optional[List[str]] = None
    wh_lex: Optional[List[str]] = None

    tense_lex: List[str] = field(default_factory=lambda: ["past", "pres"])
    asp_lex: List[str] = field(default_factory=lambda: ["prog", "perf"])
    agrs_lex: List[str] = field(default_factory=lambda: ["agrS0", "agrS1"])
    agro_lex: List[str] = field(default_factory=lambda: ["agrO0", "agrO1"])

    # fallback sizes
    n_verbs: int = 3
    n_nouns: int = 3
    n_adjs: int = 2
    n_comps: int = 2
    n_wh: int = 2

    # english preset ------------------------------------------------
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
            verb_lex=["eat", "see", "love", "give"],
            noun_lex=["tree", "horse", "dog", "cat", "apple"],
            adj_lex=["big", "small", "red", "green", "blue", "fuzzy"],
            det_lex=["the", "a"],
            comp_lex=["that"],
            wh_lex=["who", "what", "where"],
        )


def _lex(pos: str, words: List[str]) -> List[str]:
    return [f"{pos} -> '{w}'" for w in words]


def generate_cfg(p: GrammarParams) -> str:
    rules: List[str] = []

    # ----- CP / S layer -----
    rules.append("S -> CP")
    if p.wh_movement:
        rules.append("CP -> WH TP")  # root wh: no overt C
    else:
        rules.append("CP -> C TP" if p.comp_initial else "CP -> TP C")

    # ----- TP -----
    rules.append("TP -> NP_SUBJ T_BAR")
    rules.append("NP_SUBJ -> PRO | NP" if p.pro_drop else "NP_SUBJ -> NP")

    # ----- T_BAR (+AgrS) -----
    if p.rich_agreement:
        if p.verb_raise:
            rules += [
                "T_BAR -> AGR_S_NODE VP",
                "AGR_S_NODE -> T AGR_S",
            ]
        else:
            rules += [
                "T_BAR -> VP AGR_S_NODE",
                "AGR_S_NODE -> AGR_S T",
            ]
    else:
        rules.append("T_BAR -> T VP" if p.verb_raise else "T_BAR -> VP T")

    # ----- VP backbone -----
    rules += ["VP -> ASP_PHRASE", "ASP_PHRASE -> ASP little_vP"]

    # ---------------- object position & little_vP ------------------
    if p.object_shift:
        # object raises to spec‑AgrOP; V̄ lacks internal OBJ
        rules += [
            "little_vP -> AGR_OP V_BAR",
            "AGR_OP -> NP_OBJ AGR_O_NODE",
            "AGR_O_NODE -> AGR_O NP_OBJ_INNER",  # CNF helper
            "NP_OBJ -> NP",  # moved object
            "NP_OBJ_INNER -> EPS_OBJ",  # placeholder (λ‑pos)
        ]
        # V_BAR for shifted languages – no OBJ inside
        if p.rich_agreement:
            rules += [
                "V_BAR -> V_AGR",
                "V_AGR -> V AGR_O",
            ]
        else:
            rules.append("V_BAR -> V")
    else:
        # no shift ⇒ only one object inside V̄
        rules.append("little_vP -> V_BAR")
        if p.rich_agreement:
            order = "V_AGR OBJ" if p.head_initial else "OBJ V_AGR"
            rules += [f"V_BAR -> {order}", "V_AGR -> V AGR_O"]
        else:
            order = "V OBJ" if p.head_initial else "OBJ V"
            rules.append(f"V_BAR -> {order}")
        rules.append("OBJ -> NP")

    # Still need alias when object_shift = True and OBJ category used elsewhere
    if p.object_shift:
        rules.append("EPS_OBJ -> '∅'")  # pronounced null placeholder

    # ----- noun phrase -----
    if p.spec_first:
        rules += [
            "NP -> DET N_BAR",
            "N_BAR -> N",
            "N_BAR -> ADJLIST N",
        ]
    else:
        rules += [
            "NP -> N_BAR DET",
            "N_BAR -> N",
            "N_BAR -> N ADJLIST",
        ]

    rules += ["ADJLIST -> ADJ", "ADJLIST -> ADJ ADJLIST"]

    dets = p.det_lex if p.det_lex is not None else [f"det{i}" for i in range(2)]
    verbs = (
        p.verb_lex if p.verb_lex is not None else [f"verb{i}" for i in range(p.n_verbs)]
    )
    nouns = (
        p.noun_lex if p.noun_lex is not None else [f"noun{i}" for i in range(p.n_nouns)]
    )
    adjs = p.adj_lex if p.adj_lex is not None else [f"adj{i}" for i in range(p.n_adjs)]
    comps = (
        p.comp_lex if p.comp_lex is not None else [f"c{i}" for i in range(p.n_comps)]
    )
    whs = p.wh_lex if p.wh_lex is not None else [f"wh{i}" for i in range(p.n_wh)]

    rules += _lex("DET", dets)
    rules += _lex("T", [f"t_{x}" for x in p.tense_lex])
    rules += _lex("ASP", [f"asp_{x}" for x in p.asp_lex])
    if not p.wh_movement:
        rules += _lex("C", comps)
    rules += _lex("WH", whs)
    rules += _lex("V", verbs)
    rules += _lex("N", nouns)
    rules += _lex("ADJ", adjs if adjs else ["dummy_adj"])

    if p.rich_agreement:
        rules += _lex("AGR_S", p.agrs_lex)
        rules += _lex("AGR_O", p.agro_lex)
    if p.pro_drop:
        rules += _lex("PRO", ["pro"])

    grammar_str = "\n".join(rules)

    # DO NOT ATTEMPT TO DO ANYTHING WITH fg_grammar HERE
    # SIMPLY RETURN THE STRING, THAT IS ALL
    # **DO NOT EDIT THE COMMENTS**
    return grammar_str


# BEGIN FG BLOCK: CUSTOM IMPORTS. **DO NOT EDIT**
# **DO NOT EVEN EDIT THE COMMENTS**
if __name__ == "__main__":
    english_grammar = fg_grammar.Grammar.from_string(
        generate_cfg(GrammarParams.english()), grammar_type=GType.CFG
    )

    print(english_grammar.as_cfg)

    for _ in range(5):
        sample = english_grammar.generate_tree()
        print(sample["string"])
        print(sample["parse"])
# END FG BLOCK
