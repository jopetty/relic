# metaxbargrammar.py

import pprint
from dataclasses import asdict, dataclass, field
from typing import List, Literal, Optional

import formal_gym.grammar as fg_grammar

GType = fg_grammar.Grammar.Type


def _shell_rules(
    phrase: str,
    spec: str,
    head: str,
    comp: str,
    *,
    head_initial,
    spec_first,
):
    """Return the two CNF rules for a Larson shell.

    XP -> SPEC XBAR   /  XBAR SPEC
    XBAR -> X⁰ COMP   /  COMP X⁰
    """
    xbar: str = f"{head.upper()}BAR"  # unique bar label
    rules: list[str] = []

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


def _sync_shell_rules(
    phrase: str,
    spec: str,
    head: str,
    comp: str,
    *,
    head_initial_l: bool = True,
    spec_first_l: bool = True,
    head_initial_r: bool = True,
    spec_first_r: bool = True,
):
    """Return the two synchronized CNF rules for a Larson shell.

    XP -> <SPEC XBAR, SPEC XBAR>   /  <XBAR SPEC, XBAR SPEC>
    XBAR -> <X⁰ COMP, X⁰ COMP>    /  <COMP X⁰, COMP X⁰>
    """
    xbar: str = f"{head.upper()}BAR"
    rules: list[str] = []

    # specifier placement
    left_spec: str = f"{spec} {xbar}" if spec_first_l else f"{xbar} {spec}"
    right_spec: str = f"{spec} {xbar}" if spec_first_r else f"{xbar} {spec}"
    rules.append(f"{phrase} -> <{left_spec}, {right_spec}>")

    # head direction
    left_head: str = f"{head} {comp}" if head_initial_l else f"{comp} {head}"
    right_head: str = f"{head} {comp}" if head_initial_r else f"{comp} {head}"
    rules.append(f"{xbar} -> <{left_head}, {right_head}>")

    return rules


# ------------------------------------------------------------------
#  PARAMETER BUNDLE
# ------------------------------------------------------------------
@dataclass
class GrammarParams:
    # syntactic switches
    head_initial: bool = True
    spec_first: bool = True
    pro_drop: bool = False
    # verb_raise: bool = False
    # object_shift: bool = False
    # rich_agreement: bool = False
    proper_with_det: bool = False

    # lexical items
    verb_lex: Optional[List[str]] = None
    noun_lex: Optional[List[str]] = None
    propn_lex: Optional[List[str]] = None
    adj_lex: Optional[List[str]] = None
    det_lex: Optional[List[str]] = None
    comp_lex: Optional[List[str]] = None
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

    # english preset
    @classmethod
    def english(cls) -> "GrammarParams":
        return cls(
            head_initial=True,
            spec_first=True,
            pro_drop=False,
            # verb_raise=False,
            # object_shift=False,
            # rich_agreement=False,
            proper_with_det=False,
            verb_lex=["eats", "sees", "loves", "hears"],
            noun_lex=["tree", "horse", "dog", "cat", "apple"],
            propn_lex=["john", "mary", "sue", "bob"],
            adj_lex=["big", "small", "red", "green", "blue", "fuzzy", "round"],
            det_lex=["the", "a"],
            comp_lex=["that"],
        )

    @classmethod
    def german(cls) -> "GrammarParams":
        return cls(
            head_initial=False,  # German is head-final in VP
            spec_first=True,
            comp_initial=False,  # German has verb-final complement order
            wh_movement=True,
            pro_drop=True,  # German allows pro-drop more freely
            verb_raise=True,  # German has verb raising
            object_shift=False,
            rich_agreement=True,  # German has richer agreement
            proper_with_det=False,
            verb_lex=["essen", "sehen", "lieben", "geben"],
            noun_lex=["baum", "pferd", "hund", "katze", "apfel"],
            propn_lex=["johann", "maria", "susanne", "robert"],
            adj_lex=["groß", "klein", "rot", "grün", "blau", "flauschig", "rund"],
            det_lex=["der", "ein"],
            comp_lex=["dass"],
            wh_lex=["wer", "was", "wo", "wann", "warum"],
        )


@dataclass
class SyncGrammarParams:
    """Paired grammar parameters for Synchronous CFG generation."""

    left: GrammarParams
    right: GrammarParams

    def __post_init__(self):
        """Ensure lexicon sizes are compatible between left and right grammars."""
        self._align_lexicon_sizes()

    def _align_lexicon_sizes(self):
        """Align lexicon sizes between left and right grammars."""
        # Take the maximum size for each lexicon type
        max_verbs: int = max(self.left.n_verbs, self.right.n_verbs)
        max_nouns: int = max(self.left.n_nouns, self.right.n_nouns)
        max_propns: int = max(self.left.n_propns, self.right.n_propns)
        max_adjs: int = max(self.left.n_adjs, self.right.n_adjs)
        max_dets: int = max(self.left.n_dets, self.right.n_dets)
        max_comps: int = max(self.left.n_comps, self.right.n_comps)
        max_wh: int = max(self.left.n_wh, self.right.n_wh)

        # Update both grammars to have matching sizes
        self.left.n_verbs = self.right.n_verbs = max_verbs
        self.left.n_nouns = self.right.n_nouns = max_nouns
        self.left.n_propns = self.right.n_propns = max_propns
        self.left.n_adjs = self.right.n_adjs = max_adjs
        self.left.n_dets = self.right.n_dets = max_dets
        self.left.n_comps = self.right.n_comps = max_comps
        self.left.n_wh = self.right.n_wh = max_wh

        # Extend lexicons if needed
        self._extend_lexicon("verb_lex", max_verbs, "verb")
        self._extend_lexicon("noun_lex", max_nouns, "noun")
        self._extend_lexicon("propn_lex", max_propns, "name")
        self._extend_lexicon("adj_lex", max_adjs, "adj")
        self._extend_lexicon("det_lex", max_dets, "det")
        self._extend_lexicon("comp_lex", max_comps, "c")
        self._extend_lexicon("wh_lex", max_wh, "wh")

    def _extend_lexicon(self, attr_name: str, target_size: int, prefix: str):
        """Extend a lexicon to target size if needed."""
        for grammar in [self.left, self.right]:
            lex = getattr(grammar, attr_name)
            if lex is not None and len(lex) < target_size:
                # Extend with generic items
                for i in range(len(lex), target_size):
                    lex.append(f"{prefix}{i}")
                setattr(grammar, attr_name, lex)

    @classmethod
    def english_german(cls):
        """Example: English-German synchronous grammar."""
        english: GrammarParams = GrammarParams.english()
        german: GrammarParams = GrammarParams.german()
        return cls(left=english, right=german)


def _lex(pos: str, words: List[str]) -> List[str]:
    return [f"{pos} -> '{w}'" for w in words]


def _sync_lex(pos: str, left_words: List[str], right_words: List[str]) -> List[str]:
    """Generate synchronized lexical rules."""
    if len(left_words) != len(right_words):
        raise ValueError(
            f"Lexicon size mismatch for {pos}: {len(left_words)} vs {len(right_words)}"
        )

    return [f"{pos} -> <'{lw}', '{rw}'>" for lw, rw in zip(left_words, right_words)]


def generate_cfg(p: GrammarParams) -> str:
    rules: List[str] = []

    # ----- S layer: direct matrix clause, no complementizer or WH
    rules.append("S -> TP")

    # ----- TP shell -----
    rules += _shell_rules(
        phrase="TP",
        spec="NP_SUBJ",
        head="T_HEAD",
        comp="VP",
        head_initial=p.head_initial,
        spec_first=p.spec_first,
    )
    rules.append("T_HEAD -> T")

    # NP_SUBJ is either pro or a full DP
    np_subj_rule: Literal["NP_SUBJ -> PRO | DP", "NP_SUBJ -> DP"] = (
        "NP_SUBJ -> PRO | DP" if p.pro_drop else "NP_SUBJ -> DP"
    )
    rules.append(np_subj_rule)

    # ----- VP shell -----
    rules.append("VP -> V_HEAD OBJ_PHRASE")
    rules.append("V_HEAD -> V")
    # objects are full DPs
    rules.append("OBJ_PHRASE -> DP")

    # ----- DP & NP shells -----
    # DP → DET NP
    rules.append("DP -> DET NP")

    # NP → N_HEAD      (bare N‐bar)
    #    → AdjP NP     (adjoin adjectives)
    rules.append("NP -> N_HEAD")
    rules.append("NP -> AdjP NP")

    # N_HEAD → N | PROPN
    if p.proper_with_det:
        rules.append("N_HEAD -> PROPN")
    else:
        rules.append("N_HEAD -> N | PROPN")

    # AdjP projection
    rules.append("AdjP -> ADJ")

    # ----- Lexicon (no WH, no complementizer) -----
    rules += _lex("DET", p.det_lex)
    rules += _lex("T", [f"t_{x}" for x in p.tense_lex])
    rules += _lex("ASP", [f"asp_{x}" for x in p.asp_lex])
    rules += _lex("V", p.verb_lex)
    rules += _lex("N", p.noun_lex)
    rules += _lex("PROPN", p.propn_lex)
    rules += _lex("ADJ", p.adj_lex or ["dummy_adj"])

    if p.pro_drop:
        rules += _lex("PRO", ["pro"])

    return "\n".join(rules)


def generate_scfg(sp: SyncGrammarParams) -> str:
    """Generate Synchronous Context-Free Grammar rules."""
    rules: List[str] = []

    # ----- CP / S layer -----
    rules.append("S -> <CP, CP>")

    # WH‑root question rule (uses head_initial now)
    if sp.left.wh_movement or sp.right.wh_movement:
        rules += _sync_shell_rules(
            "CP",
            "WH",
            "CNULL",
            "TP",
            head_initial_l=sp.left.head_initial,
            spec_first_l=True,
            head_initial_r=sp.right.head_initial,
            spec_first_r=True,
        )

    # Declarative CP with overt C
    left_cp: Literal["C TP", "TP C"] = "C TP" if sp.left.head_initial else "TP C"
    right_cp: Literal["C TP", "TP C"] = "C TP" if sp.right.head_initial else "TP C"
    rules.append(f"CP -> <{left_cp}, {right_cp}>")

    # Declarative CP with null C
    left_cnull: Literal["CNULL TP", "TP CNULL"] = (
        "CNULL TP" if sp.left.comp_initial else "TP CNULL"
    )
    right_cnull: Literal["CNULL TP", "TP CNULL"] = (
        "CNULL TP" if sp.right.comp_initial else "TP CNULL"
    )
    rules.append(f"CP -> <{left_cnull}, {right_cnull}>")

    # ----- TP shell -----
    rules += _sync_shell_rules(
        "TP",
        "NP_SUBJ",
        "T_HEAD",
        "VP",
        head_initial_l=sp.left.verb_raise,
        spec_first_l=True,
        head_initial_r=sp.right.verb_raise,
        spec_first_r=True,
    )

    # T_HEAD rules
    left_t: Literal["T AGR_S", "AGR_S T", "T"] = (
        "T AGR_S"
        if (sp.left.rich_agreement and sp.left.verb_raise)
        else "AGR_S T"
        if sp.left.rich_agreement
        else "T"
    )
    right_t: Literal["T AGR_S", "AGR_S T", "T"] = (
        "T AGR_S"
        if (sp.right.rich_agreement and sp.right.verb_raise)
        else "AGR_S T"
        if sp.right.rich_agreement
        else "T"
    )
    rules.append(f"T_HEAD -> <{left_t}, {right_t}>")

    # Subject rules: expand any PRO | NP alternation into separate synchronous productions
    left_alts = ("PRO", "NP") if sp.left.pro_drop else ("NP",)
    right_alts = ("PRO", "NP") if sp.right.pro_drop else ("NP",)
    for ls in left_alts:
        for rs in right_alts:
            rules.append(f"NP_SUBJ -> <{ls}, {rs}>")

    # ----- VP shell -----
    rules += _sync_shell_rules(
        "VP",
        "ASP",
        "V_HEAD",
        "OBJ_PHRASE",
        head_initial_l=sp.left.head_initial,
        spec_first_l=True,
        head_initial_r=sp.right.head_initial,
        spec_first_r=True,
    )

    # V_HEAD rules
    left_v: Literal["V AGR_O", "V"] = "V AGR_O" if sp.left.rich_agreement else "V"
    right_v: Literal["V AGR_O", "V"] = "V AGR_O" if sp.right.rich_agreement else "V"
    rules.append(f"V_HEAD -> <{left_v}, {right_v}>")

    # Object rules
    if sp.left.object_shift or sp.right.object_shift:
        left_obj: Literal["EPS_OBJ", "NP"] = "EPS_OBJ" if sp.left.object_shift else "NP"
        right_obj: Literal["EPS_OBJ", "NP"] = (
            "EPS_OBJ" if sp.right.object_shift else "NP"
        )
        rules.append(f"OBJ_PHRASE -> <{left_obj}, {right_obj}>")
        if sp.left.object_shift or sp.right.object_shift:
            rules.append("EPS_OBJ -> <'∅', '∅'>")
    else:
        rules.append("OBJ_PHRASE -> <NP, NP>")

    # ----- NP shell -----
    rules += _sync_shell_rules(
        "NP",
        "DET",
        "N_HEAD",
        "ADJLIST",
        head_initial_l=False,
        spec_first_l=True,
        head_initial_r=False,
        spec_first_r=True,
    )

    # N_HEAD rules: align only identical categories when neither side forces PROPN,
    # otherwise cross‐product the allowed categories
    if not sp.left.proper_with_det and not sp.right.proper_with_det:
        # both sides allow N or PROPN but only align same category
        for cat in ("N", "PROPN"):
            rules.append(f"N_HEAD -> <{cat}, {cat}>")
    else:
        left_alts = ("PROPN",) if sp.left.proper_with_det else ("N", "PROPN")
        right_alts = ("PROPN",) if sp.right.proper_with_det else ("N", "PROPN")
        for ls in left_alts:
            for rs in right_alts:
                rules.append(f"N_HEAD -> <{ls}, {rs}>")

    rules += [
        "ADJLIST -> <ADJ, ADJ>",
        "ADJLIST -> <ADJ ADJLIST, ADJ ADJLIST>",
    ]

    # ----- Synchronized Lexicon -----
    rules += _sync_lex("DET", sp.left.det_lex, sp.right.det_lex)
    rules += _sync_lex(
        "T",
        [f"t_{x}" for x in sp.left.tense_lex],
        [f"t_{x}" for x in sp.right.tense_lex],
    )
    rules += _sync_lex(
        "ASP",
        [f"asp_{x}" for x in sp.left.asp_lex],
        [f"asp_{x}" for x in sp.right.asp_lex],
    )
    rules += _sync_lex("C", sp.left.comp_lex, sp.right.comp_lex)
    rules += _sync_lex("CNULL", ["∅_C"], ["∅_C"])
    rules += _sync_lex("WH", sp.left.wh_lex, sp.right.wh_lex)
    rules += _sync_lex("V", sp.left.verb_lex, sp.right.verb_lex)
    rules += _sync_lex("N", sp.left.noun_lex, sp.right.noun_lex)
    rules += _sync_lex("PROPN", sp.left.propn_lex, sp.right.propn_lex)

    left_adj = sp.left.adj_lex if sp.left.adj_lex else ["dummy_adj"]
    right_adj = sp.right.adj_lex if sp.right.adj_lex else ["dummy_adj"]
    rules += _sync_lex("ADJ", left_adj, right_adj)

    if sp.left.rich_agreement or sp.right.rich_agreement:
        rules += _sync_lex("AGR_S", sp.left.agrs_lex, sp.right.agrs_lex)
        rules += _sync_lex("AGR_O", sp.left.agro_lex, sp.right.agro_lex)

    if sp.left.pro_drop or sp.right.pro_drop:
        rules += _sync_lex("PRO", ["pro"], ["pro"])

    return "\n".join(rules)
