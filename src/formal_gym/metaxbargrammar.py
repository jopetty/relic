# metaxbargrammar.py

import pathlib
import random
import re
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pyrootutils

import formal_gym.grammar as fg_grammar

GType = fg_grammar.Grammar.Type

PROJECT_ROOT: pathlib.Path = pyrootutils.find_root(
    search_from=__file__, indicator=".project-root"
)

CONSONANTS = list("bcdfghjklmnpqrstvwxyz")
VOWELS = list("aeiou")


def shell_rules(
    *,
    head: str,
    spec: str,
    comp: str,
    head_initial: bool = True,
    spec_first: bool = True,
    head_initial_r: bool | None = None,
    spec_first_r: bool | None = None,
):
    """
    Generate CNF rules for a Larson shell, monolingual or synchronous.
    If head_initial_r/spec_first_r are provided (not None), emit synchronous rules.
    Otherwise, emit monolingual rules.
    The phrase label is derived as f"{head}P".
    """
    phrase: str = f"{head}P"
    xbar: str = f"{head.upper()}BAR"
    rules: list[str] = []

    if head_initial_r is not None and spec_first_r is not None:
        # Synchronous rules
        left_spec = f"{spec} {xbar}" if spec_first else f"{xbar} {spec}"
        right_spec = f"{spec} {xbar}" if spec_first_r else f"{xbar} {spec}"
        rules.append(f"{phrase} -> <{left_spec}, {right_spec}>")

        left_head = f"{head} {comp}" if head_initial else f"{comp} {head}"
        right_head = f"{head} {comp}" if head_initial_r else f"{comp} {head}"
        rules.append(f"{xbar} -> <{left_head}, {right_head}>")
    else:
        # Monolingual rules
        if spec_first:
            rules.append(f"{phrase} -> {spec} {xbar}")
        else:
            rules.append(f"{phrase} -> {xbar} {spec}")

        if head_initial:
            rules.append(f"{xbar} -> {head} {comp}")
        else:
            rules.append(f"{xbar} -> {comp} {head}")

    return rules


# ------------------------------------------------------------------
#  HELPER FUNCTIONS
# ------------------------------------------------------------------
def parse_syllable_format(template: str):
    """Parses the syllable structure template into a list of tokens

    Args:
        template: str, regex that describes the syllable structure of the language

    Returns:
        list of tokens
    """
    tokens = re.findall(r"C\*|V\*|C\?|V\?|C|V", template)
    return tokens


def generate_cluster(cluster_size: int) -> str:
    """Generates a consonant cluster

    Args:
    int cluster_size: number of chars in the cluster
    """

    sonority_hierarchy = {
        "l": 0.15,
        "m": 0.3,
        "n": 0.3,
        "v": 0.45,
        "z": 0.45,
        "f": 0.6,
        "s": 0.6,
        "b": 0.75,
        "d": 0.75,
        "g": 0.75,
        "p": 0.9,
        "t": 0.9,
        "k": 0.9,
    }

    chars = list(sonority_hierarchy.keys())
    weights = [sonority_hierarchy[c] for c in chars]
    cluster = ""

    for _ in range(cluster_size):
        c = random.choices(chars, weights=weights, k=1)[0]
        cluster = cluster + c

    return cluster


def generate_syllable(tokens: list[str], max_cons: int):
    """Generates a random syllable that conforms to the syllable structure

    Args:
        tokens: list of tokens
        max_cons: maximum number of consonants allowed in a cluster
    """

    result = []
    for token in tokens:
        if token == "C":
            result.append(random.choice(CONSONANTS))
        elif token == "V":
            result.append(random.choice(VOWELS))
        elif token == "C*":
            count = random.randint(1, max_cons)
            result.extend(generate_cluster(count))
        elif token == "V*":
            count = random.randint(1, 2)
            result.extend(random.choices(VOWELS, k=count))
        elif token == "C?":
            if random.random() < 0.5:
                result.append(random.choice(CONSONANTS))
        elif token == "V?":
            if random.random() < 0.5:
                result.append(random.choice(VOWELS))
    return "".join(result)


def sample_string(syllable_structure: list[str], avg_syllables: int, max_cons: int):
    """Generates a random string that conforms to the syllable structure

    Args:
        syllable_structure: str, regex that describes the syllable structure of the language
        avg_syllables: int, the number of syllables will be sampled from a Gaussian with this value as the mean
        max_cons: int, the maximum number of consonants that is permitted in a consonant cluster (assuming the language has C*)

    Returns:
        string
    """
    string: str = ""

    def _zero_truncated_poisson(rate: float) -> int:
        """Sample from a zero-truncated Poisson distribution."""
        u: float = np.random.uniform(np.exp(-rate), 1)
        t: float = -np.log(u)
        return 1 + np.random.poisson(rate - t)

    # Sample number of syllables from a Normal distribution
    lambda_poisson: float = avg_syllables
    num_syllables: int = _zero_truncated_poisson(lambda_poisson)
    for _ in range(num_syllables + 1):
        string = string + generate_syllable(syllable_structure, max_cons=max_cons)

    return string


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
        syllable_struct: Syllable structure of the grammar (e.g. CV?, CVC* etc.)
        avg_syllables: Average number of syllables in a word in this language
        max_consonants: Maximum number of consonants allowed in a cluster
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
    syllable_struct: str = ""
    avg_syllables: int = 2
    max_consonants: int = 2

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
    tenses: list[str] = field(default_factory=lambda: ["∅_T_pres"])
    asps: list[str] = field(default_factory=lambda: ["∅_Asp_prog"])

    def __post_init__(self):
        """Instantiates the grammar lexicon.

        If a list of strings are passed in for a particular parameter (eg 'nouns')
        then we use those; otherwise, we generate the appropriate number of lexical
        items for that parameter.
        """

        # Load syllable structures file
        syllables_file: pathlib.Path = (
            PROJECT_ROOT / "src" / "formal_gym" / "resources" / "syllables.txt"
        )
        syllables = syllables_file.read_text().splitlines()
        if self.syllable_struct is None:
            self.syllable_struct = random.choice(syllables)
        syllable_struct_tokens = parse_syllable_format(self.syllable_struct)

        # Helper to resolve int or list to list
        def resolve(val, prefix):
            if isinstance(val, int):
                return [
                    sample_string(
                        syllable_struct_tokens,
                        avg_syllables=self.avg_syllables,
                        max_cons=self.max_consonants,
                    )
                    for _ in range(val)
                ]
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
        """Generate a CFG string from the grammar parameters using GrammarRuleBuilder."""
        builder = GrammarRuleBuilder(self)
        rules = builder.build_rules()
        lexicon = builder.build_lexicon()
        return "\n".join(rules + lexicon)

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

    def as_cfg_str(self) -> str:
        """Generate Synchronous Context-Free Grammar rules using GrammarRuleBuilder."""
        builder = GrammarRuleBuilder(None, sync_params=self)
        rules = builder.build_rules()
        lexicon = builder.build_lexicon()
        return "\n".join(rules + lexicon)

    @classmethod
    def english_german(cls):
        """Example: English-German synchronous grammar."""
        english: GrammarParams = GrammarParams.english()
        german: GrammarParams = GrammarParams.german()
        return cls(left=english, right=german)

    @classmethod
    def english_spanish(cls):
        """Example: English-Spanish synchronous grammar."""
        english: GrammarParams = GrammarParams.english()
        spanish: GrammarParams = GrammarParams.spanish()
        return cls(left=english, right=spanish)


def _lex(pos: str, words: list[str]) -> list[str]:
    return [f"{pos} -> '{w}'" for w in words]


def _sync_lex(pos: str, left_words: list[str], right_words: list[str]) -> list[str]:
    """Generate synchronized lexical rules."""
    if len(left_words) != len(right_words):
        raise ValueError(
            f"Lexicon size mismatch for {pos}: {len(left_words)} vs {len(right_words)}"
        )

    return [f"{pos} -> <'{lw}', '{rw}'>" for lw, rw in zip(left_words, right_words)]


class GrammarRuleBuilder:
    """Builds grammar rules for both monolingual and synchronous grammars."""

    def __init__(self, params, sync_params=None):
        self.params = params
        self.sync_params = sync_params  # None for monolingual
        if sync_params:
            self.left = sync_params.left
            self.right = sync_params.right
        else:
            self.left = self.right = None

    def emit(self, lhs, rhs_l, rhs_r=None):
        if self.sync_params is not None:
            return f"{lhs} -> <{rhs_l}, {rhs_r}>"
        else:
            return f"{lhs} -> {rhs_l}"

    def build_rules(self):
        rules = []
        # Defensive: check left/right for sync mode
        if self.sync_params:
            if self.left is None or self.right is None:
                raise ValueError(
                    "SyncGrammarParams must have both left and right GrammarParams."
                )

        # S layer: matrix clause with null complementizer
        if self.sync_params:
            rules.append(self.emit("S", "CP_matrix", "CP_matrix"))
            rules.append(self.emit("CP_matrix", "CNULL TP", "CNULL TP"))
            rules.append(self.emit("CP_embed", "C TP", "C TP"))
        else:
            rules.append(self.emit("S", "CP_matrix"))
            rules.append(self.emit("CP_matrix", "CNULL TP"))
            rules.append(self.emit("CP_embed", "C TP"))

        # TP shell
        if self.sync_params:
            rules += shell_rules(
                head="T",
                spec="NP_SUBJ",
                comp="VP",
                head_initial=getattr(self.left, "head_initial", True),
                spec_first=getattr(self.left, "spec_first", True),
                head_initial_r=getattr(self.right, "head_initial", True),
                spec_first_r=getattr(self.right, "spec_first", True),
            )
        else:
            rules += shell_rules(
                head="T",
                spec="NP_SUBJ",
                comp="VP",
                head_initial=getattr(self.params, "head_initial", True),
                spec_first=getattr(self.params, "spec_first", True),
            )

        # Subject rules
        if self.sync_params:
            if getattr(self.left, "pro_drop", False):
                rules.append(self.emit("NP_SUBJ", "PRO", "PRO"))
            rules.append(self.emit("NP_SUBJ", "PRON", "PRON"))
            if not getattr(self.left, "proper_with_det", False):
                rules.append(self.emit("NP_SUBJ", "PROPN", "PROPN"))
            rules.append(self.emit("NP_SUBJ", "DP", "DP"))
        else:
            if getattr(self.params, "pro_drop", False):
                rules.append(self.emit("NP_SUBJ", "PRO"))
            rules.append(self.emit("NP_SUBJ", "PRON"))
            if not getattr(self.params, "proper_with_det", False):
                rules.append(self.emit("NP_SUBJ", "PROPN"))
            rules.append(self.emit("NP_SUBJ", "DP"))

        # VP shell
        if self.sync_params:
            rules.append(self.emit("VP", "V_HEAD OBJ_PHRASE", "V_HEAD OBJ_PHRASE"))
            rules.append(self.emit("V_HEAD", "V", "V"))
            rules.append(self.emit("OBJ_PHRASE", "DP", "DP"))
            rules.append(self.emit("OBJ_PHRASE", "CP_embed", "CP_embed"))
        else:
            rules.append(self.emit("VP", "V_HEAD OBJ_PHRASE"))
            rules.append(self.emit("V_HEAD", "V"))
            rules.append(self.emit("OBJ_PHRASE", "DP"))
            rules.append(self.emit("OBJ_PHRASE", "CP_embed"))

        # DP shell
        if self.sync_params:
            rules.append(self.emit("DP", "DP_def", "DP_def"))
            rules.append(self.emit("DP", "DP_indef", "DP_indef"))
            rules.append(self.emit("DP_def", "DET_def NP", "DET_def NP"))
            rules.append(self.emit("DP_indef", "DET_indef NP", "DET_indef NP"))
            left_pwd = getattr(self.left, "proper_with_det", False)
            right_pwd = getattr(self.right, "proper_with_det", False)
            if left_pwd and right_pwd:
                rules.append(self.emit("DP_def", "DET_def PROPN", "DET_def PROPN"))
            elif not left_pwd and not right_pwd:
                rules.append(self.emit("DP_def", "PROPN", "PROPN"))
        else:
            rules.append(self.emit("DP", "DP_def"))
            rules.append(self.emit("DP", "DP_indef"))
            rules.append(self.emit("DP_def", "DET_def NP"))
            rules.append(self.emit("DP_indef", "DET_indef NP"))
            if getattr(self.params, "proper_with_det", False):
                rules.append(self.emit("DP_def", "DET_def PROPN"))
            else:
                rules.append(self.emit("DP_def", "PROPN"))

        # NP rules
        if self.sync_params:
            rules.append(self.emit("NP", "N_HEAD", "N_HEAD"))
            rules.append(self.emit("NP", "AdjP NP", "AdjP NP"))
            rules.append(self.emit("NP_COMMON", "N", "N"))
            rules.append(self.emit("NP_COMMON", "AdjP NP_COMMON", "AdjP NP_COMMON"))
            rules.append(self.emit("AdjP", "ADJ", "ADJ"))
        else:
            rules.append(self.emit("NP", "N_HEAD"))
            rules.append(self.emit("NP", "AdjP NP"))
            rules.append(self.emit("NP_COMMON", "N"))
            rules.append(self.emit("NP_COMMON", "AdjP NP_COMMON"))
            rules.append(self.emit("AdjP", "ADJ"))

        # N_HEAD rules
        if self.sync_params:
            left_pwd = getattr(self.left, "proper_with_det", False)
            right_pwd = getattr(self.right, "proper_with_det", False)
            if not left_pwd and not right_pwd:
                for cat in ("N", "PROPN"):
                    rules.append(self.emit("N_HEAD", cat, cat))
            else:
                left_alts = ("PROPN",) if left_pwd else ("N", "PROPN")
                right_alts = ("PROPN",) if right_pwd else ("N", "PROPN")
                for ls in left_alts:
                    for rs in right_alts:
                        rules.append(self.emit("N_HEAD", ls, rs))
        else:
            if getattr(self.params, "proper_with_det", False):
                rules.append(self.emit("N_HEAD", "PROPN"))
            else:
                rules.append(self.emit("N_HEAD", "N | PROPN"))

        return rules

    def build_lexicon(self):
        rules = []
        if self.sync_params:
            if self.left is None or self.right is None:
                raise ValueError(
                    "SyncGrammarParams must have both left and right GrammarParams."
                )
            # Synchronized lexicon
            rules += _sync_lex(
                "DET_def",
                getattr(self.left, "det_def_lex", []),
                getattr(self.right, "det_def_lex", []),
            )
            rules += _sync_lex(
                "DET_indef",
                getattr(self.left, "det_indef_lex", []),
                getattr(self.right, "det_indef_lex", []),
            )
            rules += _sync_lex(
                "T",
                getattr(self.left, "tense_lex", []),
                getattr(self.right, "tense_lex", []),
            )
            rules += _sync_lex(
                "ASP",
                getattr(self.left, "asp_lex", []),
                getattr(self.right, "asp_lex", []),
            )
            rules += _sync_lex(
                "V",
                getattr(self.left, "verb_lex", []),
                getattr(self.right, "verb_lex", []),
            )
            rules += _sync_lex(
                "N",
                getattr(self.left, "noun_lex", []),
                getattr(self.right, "noun_lex", []),
            )
            rules += _sync_lex(
                "PROPN",
                getattr(self.left, "propn_lex", []),
                getattr(self.right, "propn_lex", []),
            )
            rules += _sync_lex(
                "PRON",
                getattr(self.left, "pron_lex", []),
                getattr(self.right, "pron_lex", []),
            )
            rules += _sync_lex(
                "ADJ",
                getattr(self.left, "adj_lex", []),
                getattr(self.right, "adj_lex", []),
            )
            rules += _sync_lex(
                "C",
                getattr(self.left, "comp_lex", []),
                getattr(self.right, "comp_lex", []),
            )
            rules.append("CNULL -> <'∅', '∅'>")
            if getattr(self.left, "pro_drop", False) or getattr(
                self.right, "pro_drop", False
            ):
                rules.append("PRO -> <'∅', '∅'>")
        else:
            # Monolingual lexicon
            rules += _lex("DET_def", list(getattr(self.params, "det_def_lex", [])))
            rules += _lex("DET_indef", list(getattr(self.params, "det_indef_lex", [])))
            rules += _lex("T", [f"{x}" for x in getattr(self.params, "tense_lex", [])])
            rules += _lex("ASP", [f"{x}" for x in getattr(self.params, "asp_lex", [])])
            rules += _lex("V", list(getattr(self.params, "verb_lex", [])))
            rules += _lex("N", list(getattr(self.params, "noun_lex", [])))
            rules += _lex("PROPN", list(getattr(self.params, "propn_lex", [])))
            rules += _lex("PRON", list(getattr(self.params, "pron_lex", [])))
            adj_lex = list(getattr(self.params, "adj_lex", []))
            rules += _lex("ADJ", adj_lex if adj_lex else ["dummy_adj"])
            rules += _lex("C", list(getattr(self.params, "comp_lex", [])))
            rules.append("CNULL -> '∅'")
            if getattr(self.params, "pro_drop", False):
                rules.append("PRO -> '∅'")
        return rules
