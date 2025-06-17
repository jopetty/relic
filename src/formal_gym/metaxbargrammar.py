# metaxbargrammar.py

from dataclasses import dataclass, field
from typing import List, Literal

import formal_gym.grammar as fg_grammar

GType = fg_grammar.Grammar.Type


def _shell_rules(
    phrase: str,
    spec: str,
    head: str,
    comp: str,
    *,
    head_initial: bool,
    spec_first: bool,
    null_spec: bool = False,
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

    if null_spec:
        rules.append(f"{spec} -> '∅'")

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
    """Grammar parameters for an x-bar style grammar.

    Attributes:
        head_initial: Whether the head is initial in the shell.
        spec_first: Whether the specifier is first in the shell.
        proper_with_det: Whether proper nouns take determiners.
        verb: List of verbs or number of verbs to generate.
        noun: List of nouns or number of nouns to generate.
        propn: List of proper nouns or number of proper nouns to generate.
        adj: List of adjectives or number of adjectives to generate.
        det_def: List of definite determiners or number of definite determiners
                to generate.
        det_indef: List of indefinite determiners or number of indefinite
            determiners to generate.
        comp: List of complementizers or number of complementizers to generate.
        tense_lex: List of tense labels or number of tense labels to generate.
        asp_lex: List of aspect labels or number of aspect labels to generate.
        agrs_lex: List of agreement labels or number of agreement labels to generate.
        agro_lex: List of agreement labels or number of agreement labels to generate.
    """

    @property
    def n_verbs(self) -> int:
        return len(self.verb_lex)

    @property
    def n_nouns(self) -> int:
        return len(self.noun_lex)

    @property
    def n_propns(self) -> int:
        return len(self.propn_lex)

    @property
    def n_adjs(self) -> int:
        return len(self.adj_lex)

    @property
    def n_det_defs(self) -> int:
        return len(self.det_def_lex)

    @property
    def n_det_indefs(self) -> int:
        return len(self.det_indef_lex)

    @property
    def n_comps(self) -> int:
        return len(self.comp_lex)

    @property
    def n_tense_lex(self) -> int:
        return len(self.tense_lex)

    @property
    def n_asp_lex(self) -> int:
        return len(self.asp_lex)

    # Parameters
    # ---------
    head_initial: bool = True
    spec_first: bool = True
    proper_with_det: bool = False

    # Lexicon
    # -------
    verbs: list[str] | int = 3
    nouns: list[str] | int = 3
    propns: list[str] | int = 3
    adjs: list[str] | int = 2
    det_def: list[str] | int = 2
    det_indef: list[str] | int = 2
    comps: list[str] | int = 2
    tenses: List[str] = field(default_factory=lambda: ["∅_T_pres"])
    asps: List[str] = field(default_factory=lambda: ["∅_Asp_prog"])

    def __post_init__(self):
        """Instantiates the grammar lexicon.

        If a list of strings are passed in for a particular parameter (eg 'nouns')
        then we use those; otherwise, we generate the appropriate number of lexical
        items for that parameter.
        """

        # Helper to resolve int or list to list
        def resolve(val, prefix):
            if isinstance(val, int):
                return [f"{prefix}{i}" for i in range(val)]
            return list(val)

        self.verb_lex = resolve(self.verbs, "verb")
        self.noun_lex = resolve(self.nouns, "noun")
        self.propn_lex = resolve(self.propns, "name")
        self.adj_lex = resolve(self.adjs, "adj")
        self.det_def_lex = resolve(self.det_def, "det_def")
        self.det_indef_lex = resolve(self.det_indef, "det_indef")
        self.comp_lex = resolve(self.comps, "c")
        self.tense_lex = resolve(self.tenses, "tense")
        self.asp_lex = resolve(self.asps, "asp")

    def as_cfg_str(self) -> str:
        """Generate a CFG string from the grammar parameters."""
        rules: List[str] = []

        # ----- S layer: matrix clause with null complementizer -----
        rules.append("S -> CP_matrix")
        rules.append("CP_matrix -> CNULL TP")

        # ----- Embedded CP for object clauses (with overt complementizer) -----
        rules.append("CP_embed -> C TP")

        # ----- TP shell -----
        rules += _shell_rules(
            phrase="TP",
            spec="NP_SUBJ",
            head="T",
            comp="VP",
            head_initial=self.head_initial,
            spec_first=self.spec_first,
        )

        # NP_SUBJ: subjects
        #  if proper_with_det=False, also allow bare proper nouns
        if not self.proper_with_det:
            rules.append("NP_SUBJ -> PROPN")
        rules.append("NP_SUBJ -> DP")

        # ----- VP shell -----
        rules.append("VP -> V_HEAD OBJ_PHRASE")
        rules.append("V_HEAD -> V")
        # objects can be full DPs or embedded CPs
        rules.append("OBJ_PHRASE -> DP")
        rules.append("OBJ_PHRASE -> CP_embed")

        # ----- DP shell -----
        rules.append("DP -> DP_def | DP_indef")
        rules.append("DP_def -> DET_def NP")
        rules.append("DP_indef -> DET_indef NP")
        if self.proper_with_det:
            rules.append("DP_def -> DET_def PROPN")
        else:
            rules.append("DP_def -> PROPN")

        # NP → N_HEAD          (bare N‐bar; includes PROPN or N)
        #    → AdjP NP         (adjoin adjectives)
        rules.append("NP -> N_HEAD")
        rules.append("NP -> AdjP NP")

        # N_HEAD → N | PROPN
        if self.proper_with_det:
            rules.append("N_HEAD -> PROPN")
        else:
            rules.append("N_HEAD -> N | PROPN")

        # AdjP projection
        rules.append("AdjP -> ADJ")

        # NP_COMMON (no PROPN) for indefinite DPs
        rules.append("NP_COMMON -> N")
        rules.append("NP_COMMON -> AdjP NP_COMMON")

        # ----- Lexicon (including complementizers) -----
        rules += _lex("DET_def", list(self.det_def_lex))
        rules += _lex("DET_indef", list(self.det_indef_lex))
        rules += _lex("T", [f"{x}" for x in self.tense_lex])
        rules += _lex("ASP", [f"{x}" for x in self.asp_lex])
        rules += _lex("V", list(self.verb_lex))
        rules += _lex("N", list(self.noun_lex))
        rules += _lex("PROPN", list(self.propn_lex))
        rules += _lex("ADJ", list(self.adj_lex) if self.adj_lex else ["dummy_adj"])
        rules += _lex("C", list(self.comp_lex))
        rules.append("CNULL -> '∅'")

        return "\n".join(rules)

    @classmethod
    def english(cls) -> "GrammarParams":
        """English grammar parameters."""
        return cls(
            head_initial=True,
            spec_first=True,
            proper_with_det=False,
            verbs=["eats", "sees", "loves", "hears"],
            nouns=["tree", "horse", "dog", "cat", "apple"],
            propns=["john", "mary", "sue", "bob"],
            adjs=["big", "small", "red", "green", "blue", "fuzzy", "round"],
            det_def=["the"],
            det_indef=["a"],
            comps=["that"],
        )

    @classmethod
    def german(cls) -> "GrammarParams":
        """German grammar parameters."""
        return cls(
            head_initial=False,  # German is head-final in VP
            spec_first=True,
            proper_with_det=False,
            verbs=["essen", "sehen", "lieben", "geben"],
            nouns=["baum", "pferd", "hund", "katze", "apfel"],
            propns=["johann", "maria", "susanne", "robert"],
            adjs=["groß", "klein", "rot", "grün", "blau", "flauschig", "rund"],
            det_def=["der"],
            det_indef=["ein"],
            comps=["dass"],
        )


@dataclass
class SyncGrammarParams:
    """Paired grammar parameters for Synchronous CFG generation."""

    left: GrammarParams
    right: GrammarParams

    def _align_lexicon_sizes(self):
        """Align lexicon sizes between left and right grammars."""
        max_verbs: int = max(self.left.n_verbs, self.right.n_verbs)
        max_nouns: int = max(self.left.n_nouns, self.right.n_nouns)
        max_propns: int = max(self.left.n_propns, self.right.n_propns)
        max_adjs: int = max(self.left.n_adjs, self.right.n_adjs)
        max_det_defs: int = max(self.left.n_det_defs, self.right.n_det_defs)
        max_det_indefs: int = max(self.left.n_det_indefs, self.right.n_det_indefs)
        max_comps: int = max(self.left.n_comps, self.right.n_comps)

        self._extend_lexicon("verbs", max_verbs, "verb")
        self._extend_lexicon("nouns", max_nouns, "noun")
        self._extend_lexicon("propns", max_propns, "name")
        self._extend_lexicon("adjs", max_adjs, "adj")
        self._extend_lexicon("det_defs", max_det_defs, "det")
        self._extend_lexicon("det_indefs", max_det_indefs, "det")
        self._extend_lexicon("comps", max_comps, "c")

    def _extend_lexicon(self, attr_name: str, target_size: int, prefix: str):
        """Extend a lexicon to target size if needed."""
        for grammar in [self.left, self.right]:
            lex = getattr(grammar, attr_name)
            if len(lex) < target_size:
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
    rules += _sync_lex("DET", sp.left.det_def, sp.right.det_def)
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
    rules += _sync_lex("C", sp.left.comp, sp.right.comp)
    rules += _sync_lex("CNULL", ["∅_C"], ["∅_C"])
    rules += _sync_lex("WH", sp.left.wh_lex, sp.right.wh_lex)
    rules += _sync_lex("V", sp.left.verb, sp.right.verb)
    rules += _sync_lex("N", sp.left.noun, sp.right.noun)
    rules += _sync_lex("PROPN", sp.left.propn, sp.right.propn)

    left_adj = sp.left.adj if sp.left.adj else ["dummy_adj"]
    right_adj = sp.right.adj if sp.right.adj else ["dummy_adj"]
    rules += _sync_lex("ADJ", left_adj, right_adj)

    if sp.left.rich_agreement or sp.right.rich_agreement:
        rules += _sync_lex("AGR_S", sp.left.agrs_lex, sp.right.agrs_lex)
        rules += _sync_lex("AGR_O", sp.left.agro_lex, sp.right.agro_lex)

    if sp.left.pro_drop or sp.right.pro_drop:
        rules += _sync_lex("PRO", ["pro"], ["pro"])

    return "\n".join(rules)
