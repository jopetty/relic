"""Sample a random X-bar style grammar."""

from dataclasses import dataclass, field
from typing import List, Optional

import formal_gym.grammar as fg_grammar

GType = fg_grammar.Grammar.Type


@dataclass
class GrammarParams:
    # syntactic switches
    head_initial: bool = True  # head precedes complement (english = True)
    spec_first: bool = True  # specifier precedes phrase (english = True)
    comp_initial: bool = True  # complementizer precedes TP (english “that S”)
    wh_movement: bool = True  # wh raises to Spec‑CP (english = True)
    pro_drop: bool = False  # no null subject in english
    verb_raise: bool = False  # main verb stays low in english
    object_shift: bool = False  # english lacks scandinavian object shift
    rich_agreement: bool = False  # poor agreement morphology in english

    # lexical *lists* (override numeric defaults when given)
    verb_lex: Optional[List[str]] = None
    noun_lex: Optional[List[str]] = None
    adj_lex: Optional[List[str]] = None
    det_lex: Optional[List[str]] = None
    comp_lex: Optional[List[str]] = None
    wh_lex: Optional[List[str]] = None

    # functional morphemes always get defaults (can override)
    tense_lex: List[str] = field(default_factory=lambda: ["past", "pres"])
    asp_lex: List[str] = field(default_factory=lambda: ["prog", "perf"])
    agrs_lex: List[str] = field(default_factory=lambda: ["agrS0", "agrS1"])
    agro_lex: List[str] = field(default_factory=lambda: ["agrO0", "agrO1"])

    # fallback inventory sizes (used only if corresponding *_lex is None)
    n_verbs: int = 3
    n_nouns: int = 3
    n_adjs: int = 2
    n_comps: int = 2
    n_wh: int = 2

    # ---- convenience constructor for english‑like settings ----
    @classmethod
    def english(cls):
        """preset params + ordinary english lexicon"""
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
            noun_lex=["john", "mary", "dog", "cat", "apple"],
            adj_lex=["big", "small"],
            det_lex=["the", "a"],
            comp_lex=["that"],
            wh_lex=["who", "what", "where"],
        )


def _lex(pos: str, words: List[str]) -> List[str]:
    return [f"{pos} -> '{w}'" for w in words]


def generate_cfg(p: GrammarParams) -> str:
    rules: List[str] = []

    # ----- cp layer -----
    rules.append("S -> CP")
    rules.append("CP -> WH CTP" if p.wh_movement else "CP -> CTP")
    rules.append("CTP -> C TP" if p.comp_initial else "CTP -> TP C")

    # ----- tp -----
    rules.append("TP -> NP_SUBJ T_BAR")
    rules.append("NP_SUBJ -> PRO | NP" if p.pro_drop else "NP_SUBJ -> NP")

    # T + agreement morphology
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

    # ----- vp spine -----
    rules += ["VP -> ASP_PHRASE", "ASP_PHRASE -> ASP little_vP"]

    if p.object_shift:
        rules += [
            "little_vP -> AGR_OP V_BAR",
            "AGR_OP -> NP_OBJ AGR_O_NODE",
            "AGR_O_NODE -> AGR_O OBJ",
        ]
    else:
        rules.append("little_vP -> NP_OBJ V_BAR")

    if p.rich_agreement:
        if p.object_shift:
            rules.append("V_BAR -> OBJ V_AGR")
        else:
            order = "OBJ V_AGR" if p.head_initial else "V_AGR OBJ"
            rules.append(f"V_BAR -> {order}")
        rules.append("V_AGR -> V AGR_O")
    else:
        order = "V OBJ" if p.head_initial else "OBJ V"
        rules.append(f"V_BAR -> {order}")

    rules += ["OBJ -> NP", "NP_OBJ -> NP"]

    # ----- noun phrase -----
    if p.spec_first:
        rules += [
            "NP -> DET N_BAR",
            "N_BAR -> N",
            "N_BAR -> N ADJLIST",
        ]
    else:
        rules += [
            "NP -> N_BAR DET",
            "N_BAR -> N",
            "N_BAR -> ADJLIST N",
        ]

    rules += ["ADJLIST -> ADJ", "ADJLIST -> ADJ ADJLIST"]

    # ----------------------------------------------------------------
    #   LEXICON  (use provided lists, else fall back to dummy items)
    # ----------------------------------------------------------------

    dets = p.det_lex if p.det_lex is not None else ["det" + str(i) for i in range(2)]
    verbs = (
        p.verb_lex
        if p.verb_lex is not None
        else ["verb" + str(i) for i in range(p.n_verbs)]
    )
    nouns = (
        p.noun_lex
        if p.noun_lex is not None
        else ["noun" + str(i) for i in range(p.n_nouns)]
    )
    adjs = (
        p.adj_lex
        if p.adj_lex is not None
        else ["adj" + str(i) for i in range(p.n_adjs)]
    )
    comps = (
        p.comp_lex
        if p.comp_lex is not None
        else ["c" + str(i) for i in range(p.n_comps)]
    )
    whs = p.wh_lex if p.wh_lex is not None else ["wh" + str(i) for i in range(p.n_wh)]

    rules += _lex("DET", dets)
    rules += _lex("T", [f"t_{x}" for x in p.tense_lex])
    rules += _lex("ASP", [f"asp_{x}" for x in p.asp_lex])
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

    return "\n".join(rules)


if __name__ == "__main__":
    english_grammar = fg_grammar.Grammar.from_string(
        generate_cfg(GrammarParams.english()), grammar_type=GType.CFG
    )

    print(english_grammar.as_cfg)

    for i in range(10):
        print(english_grammar.generate())
