# metaxbargrammar.py

from dataclasses import dataclass, field
from typing import Any, List, Literal, Optional

import nltk

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
        pro_drop: Whether to allow pro-drop (null pronominal subject).
        proper_with_det: Whether proper nouns take determiners.
        verb: List of verbs or number of verbs to generate.
        noun: List of nouns or number of nouns to generate.
        propn: List of proper nouns or number of proper nouns to generate.
        pron: List of pronouns or number of pronouns to generate.
        adj: List of adjectives or number of adjectives to generate.
        det_def: List of definite determiners or number of definite determiners
                to generate.
        det_indef: List of indefinite determiners or number of indefinite
            determiners to generate.
        comp: List of complementizers or number of complementizers to generate.
        tenses: List of tense labels or number of tense labels to generate.
        asps: List of aspect labels or number of aspect labels to generate.
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
    pro_drop: bool = False
    proper_with_det: bool = False

    # Lexicon
    # -------
    verbs: list[str] | int = 3
    nouns: list[str] | int = 3
    propns: list[str] | int = 3
    prons: list[str] | int = 2
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
        self.pron_lex = resolve(self.prons, "pron")
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
        #  if pro_drop, allow PRO too
        #  if proper_with_det=False, also allow bare proper nouns
        if self.pro_drop:
            rules.append("NP_SUBJ -> PRO")
        rules.append("NP_SUBJ -> PRON")
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
        rules.append("DP -> <DP_def, DP_def>")
        rules.append("DP -> <DP_indef, DP_indef>")
        rules.append("DP_def -> <DET_def NP, DET_def NP>")
        rules.append("DP_indef -> <DET_indef NP, DET_indef NP>")
        if self.proper_with_det:
            rules.append("DP_def -> <DET_def PROPN, DET_def PROPN>")
        else:
            rules.append("DP_def -> <PROPN, PROPN>")

        # NP rules
        rules.append("NP -> <N_HEAD, N_HEAD>")
        rules.append("NP -> <AdjP NP, AdjP NP>")
        rules.append("NP_COMMON -> <N, N>")
        rules.append("NP_COMMON -> <AdjP NP_COMMON, AdjP NP_COMMON>")
        rules.append("AdjP -> <ADJ, ADJ>")

        # N_HEAD rules
        if self.proper_with_det:
            rules.append("N_HEAD -> PROPN")
        else:
            rules.append("N_HEAD -> N | PROPN")

        # ----- Lexicon (including complementizers) -----
        rules += _lex("DET_def", list(self.det_def_lex))
        rules += _lex("DET_indef", list(self.det_indef_lex))
        rules += _lex("T", [f"{x}" for x in self.tense_lex])
        rules += _lex("ASP", [f"{x}" for x in self.asp_lex])
        rules += _lex("V", list(self.verb_lex))
        rules += _lex("N", list(self.noun_lex))
        rules += _lex("PROPN", list(self.propn_lex))
        rules += _lex("PRON", list(self.pron_lex))
        rules += _lex("ADJ", list(self.adj_lex) if self.adj_lex else ["dummy_adj"])
        rules += _lex("C", list(self.comp_lex))
        rules.append("CNULL -> '∅'")
        if self.pro_drop:
            rules.append("PRO -> '∅'")

        return "\n".join(rules)

    @classmethod
    def english(cls) -> "GrammarParams":
        """English grammar parameters."""
        return cls(
            head_initial=True,
            spec_first=True,
            pro_drop=False,
            proper_with_det=False,
            verbs=["eats", "sees", "loves", "hears"],
            nouns=["tree", "horse", "dog", "cat", "apple"],
            propns=["john", "mary", "sue", "bob"],
            prons=["he", "she", "they", "it"],
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
            pro_drop=False,  # Set to False for now; can be changed if needed
            proper_with_det=False,
            verbs=["essen", "sehen", "lieben", "geben"],
            nouns=["baum", "pferd", "hund", "katze", "apfel"],
            propns=["johann", "maria", "susanne", "robert"],
            prons=["er", "sie", "es", "wir"],
            adjs=["groß", "klein", "rot", "grün", "blau", "flauschig", "rund"],
            det_def=["der"],
            det_indef=["ein"],
            comps=["dass"],
        )

    @classmethod
    def spanish(cls) -> "GrammarParams":
        """Spanish grammar parameters."""
        return cls(
            head_initial=True,
            spec_first=True,
            pro_drop=True,
            proper_with_det=False,
            verbs=["come", "ve", "ama", "escucha"],
            nouns=["árbol", "caballo", "perro", "gato", "manzana"],
            propns=["juan", "maría", "susana", "roberto"],
            prons=["él", "ella", "ellos", "ellas"],
            adjs=["grande", "pequeño", "rojo", "verde", "azul", "suave", "redondo"],
            det_def=["el"],
            det_indef=["un"],
            comps=["que"],
        )


class XBarGrammar(fg_grammar.Grammar):
    @classmethod
    def from_params(cls, params: GrammarParams) -> "XBarGrammar":
        """Create a grammar from parameters."""
        return cls.from_string(params.as_cfg_str(), grammar_type=GType.CFG)

    def generate_tree(self, **kwargs) -> dict[str, Any]:
        """Generate a tree from the grammar."""
        gen_dict: dict[str, Any] = super().generate_tree(**kwargs)
        # add a "phonetic_string" field which takes the generated string
        # and removes every word beginning with the null symbol (∅)
        phonetic_string: str = " ".join(
            [w for w in gen_dict["string"].split(" ") if not w.startswith("∅")]
        )
        gen_dict["phonetic_string"] = phonetic_string
        return gen_dict

    def generate_with_path(
        self, max_depth: int = 50
    ) -> Optional[tuple[dict[str, Any], list[tuple[str, str]]]]:
        """Generate a tree and its derivation path.

        Returns:
            A tuple of (generation_dict, derivation_path) where:
            - generation_dict contains the same fields as generate_tree()
            - derivation_path is a list of (lhs, rhs) tuples showing the derivation
            Returns None if generation fails
        """
        derivation_path: list[tuple[str, str]] = []

        def _sample_recursive(
            symbol: nltk.Nonterminal, depth: int
        ) -> Optional[list[str]]:
            if depth > max_depth:
                return None

            # Choose a production
            prod = self._choose_production(symbol, depth, max_depth)
            if prod is None:
                return None

            # Record the derivation step
            derivation_path.append(
                (str(prod.lhs()), " ".join(str(s) for s in prod.rhs()))
            )

            # If lexical, return the terminal
            if prod.is_lexical():
                return [str(prod.rhs()[0])]

            # Otherwise, recursively generate from the RHS
            result: list[str] = []
            for rhs_sym in prod.rhs():
                if isinstance(rhs_sym, nltk.Nonterminal):
                    sub_result = _sample_recursive(rhs_sym, depth + 1)
                    if sub_result is None:
                        return None
                    result.extend(sub_result)
                else:
                    result.append(str(rhs_sym))
            return result

        try:
            # Generate the string
            result = _sample_recursive(self.as_cfg.start(), 0)
            if result is None:
                return None

            # Create the generation dict
            gen_dict = {
                "string": " ".join(result),
                "phonetic_string": " ".join(
                    [w for w in result if not w.startswith("∅")]
                ),
                "tree": None,  # We don't track the tree in this version
            }

            return gen_dict, derivation_path
        except RecursionError:
            return None

    def generate_from_path(
        self, derivation_path: list[tuple[str, str]], max_depth: int = 50
    ) -> Optional[dict[str, Any]]:
        """Generate a string following a given derivation path.

        Args:
            derivation_path: List of (lhs, rhs) tuples showing the derivation
            max_depth: Maximum depth for generation

        Returns:
            A dictionary with the same fields as generate_tree(), or None if generation fails
        """

        def _sample_from_path(
            symbol: nltk.Nonterminal, depth: int, path_idx: int
        ) -> Optional[list[str]]:
            if depth > max_depth or path_idx >= len(derivation_path):
                return None

            # Get the expected derivation step
            expected_lhs, expected_rhs = derivation_path[path_idx]
            if str(symbol) != expected_lhs:
                return None

            # Find a matching production
            for prod in self.productions_by_lhs[symbol]:
                if " ".join(str(s) for s in prod.rhs()) == expected_rhs:
                    # If lexical, return the terminal
                    if prod.is_lexical():
                        return [str(prod.rhs()[0])]

                    # Otherwise, recursively generate from the RHS
                    result: list[str] = []
                    for rhs_sym in prod.rhs():
                        if isinstance(rhs_sym, nltk.Nonterminal):
                            sub_result = _sample_from_path(
                                rhs_sym, depth + 1, path_idx + 1
                            )
                            if sub_result is None:
                                return None
                            result.extend(sub_result)
                        else:
                            result.append(str(rhs_sym))
                    return result

            return None

        try:
            # Generate the string
            result = _sample_from_path(self.as_cfg.start(), 0, 0)
            if result is None:
                return None

            # Create the generation dict
            return {
                "string": " ".join(result),
                "phonetic_string": " ".join(
                    [w for w in result if not w.startswith("∅")]
                ),
                "tree": None,  # We don't track the tree in this version
            }
        except RecursionError:
            return None


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
        max_prons: int = max(len(self.left.pron_lex), len(self.right.pron_lex))
        max_adjs: int = max(self.left.n_adjs, self.right.n_adjs)
        max_det_defs: int = max(self.left.n_det_defs, self.right.n_det_defs)
        max_det_indefs: int = max(self.left.n_det_indefs, self.right.n_det_indefs)
        max_comps: int = max(self.left.n_comps, self.right.n_comps)

        self._extend_lexicon("verbs", max_verbs, "verb")
        self._extend_lexicon("nouns", max_nouns, "noun")
        self._extend_lexicon("propns", max_propns, "name")
        self._extend_lexicon("prons", max_prons, "pron")
        self._extend_lexicon("adjs", max_adjs, "adj")
        self._extend_lexicon("det_defs", max_det_defs, "det")
        self._extend_lexicon("det_indefs", max_det_indefs, "det")
        self._extend_lexicon("comps", max_comps, "c")

    def _extend_lexicon(self, attr_name: str, target_size: int, prefix: str):
        """Extend a lexicon to target size if needed."""
        for grammar in [self.left, self.right]:
            lex = getattr(grammar, f"{attr_name}_lex")
            if len(lex) < target_size:
                # Extend with generic items
                for i in range(len(lex), target_size):
                    lex.append(f"{prefix}{i}")
                setattr(grammar, f"{attr_name}_lex", lex)

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

    # ----- S layer: matrix clause with null complementizer -----
    rules.append("S -> <CP_matrix, CP_matrix>")
    rules.append("CP_matrix -> <CNULL TP, CNULL TP>")

    # ----- Embedded CP for object clauses (with overt complementizer) -----
    rules.append("CP_embed -> <C TP, C TP>")

    # ----- TP shell -----
    rules += _sync_shell_rules(
        "TP",
        "NP_SUBJ",
        "T",
        "VP",
        head_initial_l=sp.left.head_initial,
        spec_first_l=sp.left.spec_first,
        head_initial_r=sp.right.head_initial,
        spec_first_r=sp.right.spec_first,
    )

    # Subject rules - match standalone CFG structure
    if sp.left.pro_drop:
        rules.append("NP_SUBJ -> <PRO, PRO>")
    rules.append("NP_SUBJ -> <PRON, PRON>")
    if not sp.left.proper_with_det:
        rules.append("NP_SUBJ -> <PROPN, PROPN>")
    rules.append("NP_SUBJ -> <DP, DP>")

    # ----- VP shell -----
    rules.append("VP -> <V_HEAD OBJ_PHRASE, V_HEAD OBJ_PHRASE>")
    rules.append("V_HEAD -> <V, V>")
    # objects can be full DPs or embedded CPs
    rules.append("OBJ_PHRASE -> <DP, DP>")
    rules.append("OBJ_PHRASE -> <CP_embed, CP_embed>")

    # ----- DP shell -----
    rules.append("DP -> <DP_def, DP_def>")
    rules.append("DP -> <DP_indef, DP_indef>")
    rules.append("DP_def -> <DET_def NP, DET_def NP>")
    rules.append("DP_indef -> <DET_indef NP, DET_indef NP>")
    if sp.left.proper_with_det and sp.right.proper_with_det:
        rules.append("DP_def -> <DET_def PROPN, DET_def PROPN>")
    elif not sp.left.proper_with_det and not sp.right.proper_with_det:
        rules.append("DP_def -> <PROPN, PROPN>")

    # NP rules
    rules.append("NP -> <N_HEAD, N_HEAD>")
    rules.append("NP -> <AdjP NP, AdjP NP>")
    rules.append("NP_COMMON -> <N, N>")
    rules.append("NP_COMMON -> <AdjP NP_COMMON, AdjP NP_COMMON>")
    rules.append("AdjP -> <ADJ, ADJ>")

    # N_HEAD rules
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
    rules += _sync_lex("DET_def", sp.left.det_def_lex, sp.right.det_def_lex)
    rules += _sync_lex("DET_indef", sp.left.det_indef_lex, sp.right.det_indef_lex)
    rules += _sync_lex("T", sp.left.tense_lex, sp.right.tense_lex)
    rules += _sync_lex("ASP", sp.left.asp_lex, sp.right.asp_lex)
    rules += _sync_lex("V", sp.left.verb_lex, sp.right.verb_lex)
    rules += _sync_lex("N", sp.left.noun_lex, sp.right.noun_lex)
    rules += _sync_lex("PROPN", sp.left.propn_lex, sp.right.propn_lex)
    rules += _sync_lex("PRON", sp.left.pron_lex, sp.right.pron_lex)
    rules += _sync_lex("ADJ", sp.left.adj_lex, sp.right.adj_lex)
    rules += _sync_lex("C", sp.left.comp_lex, sp.right.comp_lex)
    rules.append("CNULL -> <'∅', '∅'>")

    if sp.left.pro_drop or sp.right.pro_drop:
        rules.append("PRO -> <'∅', '∅'>")

    return "\n".join(rules)
