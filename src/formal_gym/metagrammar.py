"""Sample from grammar space."""

import collections
import datetime
import pathlib
import random
from typing import Any, NamedTuple, Sequence

import pyrootutils

import formal_gym.grammar as fg_grammar
import formal_gym.utils.utils as fg_utils

GType = fg_grammar.Grammar.Type


class Terminal(str):
    def __repr__(self):
        return "Terminal(" + super().__repr__() + ")"

    def __str__(self):
        return f"'{super().__str__()}'"


class Nonterminal(str):
    def __repr__(self):
        return "Nonterminal(" + super().__repr__() + ")"


Symbol = Terminal | Nonterminal


class Production(NamedTuple):
    lhs: Nonterminal
    rhs: tuple[Symbol]

    def __str__(self):
        return f"{self.lhs} -> {' '.join(map(str, self.rhs))}"

    @property
    def length(self):
        return len(self.rhs)

    @property
    def is_lexical(self):
        return self.length == 1


START_SYMBOL = Nonterminal("S")

PROJECT_ROOT = path = pyrootutils.find_root(
    search_from=__file__, indicator=".project-root"
)

log = fg_utils.get_logger(__name__)


def get_symbols(n_terminals: int, n_nonterminals: int) -> dict[str, list[Symbol]]:
    terminals = [Terminal(f"t{i}") for i in range(n_terminals)]
    nonterminals = [START_SYMBOL] + [
        Nonterminal(f"NT{i}") for i in range(n_nonterminals)
    ]
    return {"terminals": terminals, "nonterminals": nonterminals}


def prod_reachable(
    prod: Production, terminals: Sequence[Terminal], coreachable: set[Production]
) -> bool:
    for symbol in prod.rhs:
        if (symbol not in terminals) and (symbol not in coreachable):
            return False
    return True


def compute_coreachable_set(
    productions: Sequence[Production],
    nonterminals: Sequence[Nonterminal],
    terminals: Sequence[Terminal],
) -> set[Nonterminal]:
    coreachable = set()
    iteration = 0

    prodmap = collections.defaultdict(list)
    for prod in productions:
        prodmap[prod.lhs].append(prod)
    remaining = set(nonterminals)

    done_this_loop = 0
    while iteration == 0 or done_this_loop > 0:
        iteration += 1
        done_this_loop = 0
        for nt in remaining:
            for prod in prodmap[nt]:
                if prod_reachable(
                    prod=prod, terminals=terminals, coreachable=coreachable
                ):
                    done_this_loop += 1
                    coreachable.add(nt)
                    break
        remaining = remaining - coreachable

    return coreachable


def compute_coreachable_productions(
    productions: Sequence[Production],
    coreachable_nts: set[Nonterminal],
    terminals: Sequence[Terminal],
) -> set[Production]:
    good_productions = set()
    for prod in productions:
        for symbol in prod.rhs:
            if (symbol not in coreachable_nts) and (symbol not in terminals):
                break
        else:
            good_productions.add(prod)
    return good_productions


def compute_usable_prods(
    trim_set: set[Nonterminal],
    productions: Sequence[Production],
) -> Sequence[Production]:
    tp = []
    for prod in productions:
        if prod.lhs in trim_set:
            if prod.is_lexical or all([symbol in trim_set for symbol in prod.rhs]):
                tp.append(prod)
    return tp


def compute_trim_set(
    productions: Sequence[Production],
    nonterminals: Sequence[Nonterminal],
    terminals: Sequence[Terminal],
) -> set[Nonterminal]:
    coreachable = compute_coreachable_set(
        productions=productions, nonterminals=nonterminals, terminals=terminals
    )
    trim = set()
    good_productions = compute_coreachable_productions(
        productions=productions, coreachable_nts=coreachable, terminals=terminals
    )

    if START_SYMBOL in coreachable:
        trim.add(START_SYMBOL)
    done = len(trim)
    while done > 0:
        done = 0
        for prod in good_productions:
            if prod.lhs in trim:
                for symbol in prod.rhs:
                    if (symbol in nonterminals) and (symbol not in trim):
                        done += 1
                        trim.add(symbol)
    return trim


def load_grammar(
    productions: Sequence[Production],
    filepath: pathlib.Path,
    save_grammar: bool = True,
) -> dict[str, Any]:
    """
    Save a grammar to a file and load it back.

    Args:
        productions: List of productions.
        filepath: Path to save the grammar.
    """

    start_productions = [p for p in productions if p.lhs == START_SYMBOL]
    nonterminal_productions = [
        p for p in productions if (p.lhs != START_SYMBOL) and (not p.is_lexical)
    ]
    lexical_productions = [
        p for p in productions if (p.lhs != START_SYMBOL) and p.is_lexical
    ]

    sorted_productions = (
        start_productions + nonterminal_productions + lexical_productions
    )

    filepath.parent.mkdir(parents=True, exist_ok=True)

    grammar_string = "\n".join([f"{p}" for p in sorted_productions])

    grammar_dict = {
        "grammar": fg_grammar.Grammar.from_string(
            grammar_string, grammar_type=GType.CFG
        ),
    }

    if save_grammar:
        with open(filepath, "w") as f:
            f.write(grammar_string)
        log.info(f"Grammar saved to {filepath}")
        grammar_dict |= {"filepath": filepath}

    return grammar_dict


def sample_cfg_uniform(
    n_terminals: int,
    n_nonterminals: int,
    data_dir: pathlib.Path = PROJECT_ROOT / "data",
    name: str = "sample_uniform",
    lp: float = 0.5,
    bp: float = 0.5,
) -> dict[str, Any]:
    """
    Samples a CFG with Bernoulli probabilities for lexical and binary productions.

    Args:
        n_terminals: Number of terminals.
        n_nonterminals: Number of nonterminals.
        data_dir: Directory to save the CFG.
        name: Name of the CFG.
        lp: Lexical production probability.
        bp: Binary production probability.
    """
    assert n_terminals > 0, "Number of terminals must be greater than 0"
    assert n_nonterminals > 0, "Number of nonterminals must be greater than 0"
    assert 0 <= lp <= 1, "Lexical production probability must be between 0 and 1"
    assert 0 <= bp <= 1, "Binary production probability must be between 0 and 1"
    log.info(
        f"Sampling CFG with {n_terminals} terminals and {n_nonterminals} nonterminals"
    )
    symbol_dict = get_symbols(n_terminals, n_nonterminals)
    terminals = symbol_dict["terminals"]
    nonterminals = symbol_dict["nonterminals"]
    productions = []
    for a in nonterminals[1:]:
        for b in terminals:
            if random.random() < lp:
                productions.append(Production(lhs=a, rhs=(b,)))
    for a in nonterminals:
        for b in nonterminals[1:]:
            for c in nonterminals[1:]:
                if random.random() < bp:
                    productions.append(Production(lhs=a, rhs=(b, c)))
    filename = f"{name}_{datetime.datetime.now().strftime('%Y%m%d%H%M%S')}.cfg"
    filepath = data_dir / filename
    sampled_grammar = load_grammar(productions, filepath)
    return sampled_grammar


def sample_cfg_full(
    n_terminals: int,
    n_nonterminals: int,
    data_dir: pathlib.Path = PROJECT_ROOT / "data",
    name: str = "sample_full",
) -> dict[str, Any]:
    """Samples a CFG with all possible productions.

    Args:
        n_terminals: Number of terminals.
        n_nonterminals: Number of nonterminals.
        data_dir: Directory to save the CFG.
        name: Name of the CFG.
    """
    assert n_terminals > 0, "Number of terminals must be greater than 0"
    assert n_nonterminals > 0, "Number of nonterminals must be greater than 0"
    log.info(
        f"Sampling CFG with {n_terminals} terminals and {n_nonterminals} nonterminals"
    )
    symbol_dict = get_symbols(n_terminals, n_nonterminals)
    terminals = symbol_dict["terminals"]
    nonterminals = symbol_dict["nonterminals"]
    productions = []
    for a in nonterminals[1:]:
        for b in terminals:
            productions.append(Production(lhs=a, rhs=(b,)))
    for a in nonterminals:
        for b in nonterminals[1:]:
            for c in nonterminals[1:]:
                productions.append(Production(lhs=a, rhs=(b, c)))
    filename = f"{name}_{datetime.datetime.now().strftime('%Y%m%d%H%M%S')}.cfg"
    filepath = data_dir / filename
    sampled_grammar = load_grammar(productions, filepath)
    return sampled_grammar


def sample_cfg_raw(
    n_terminals: int,
    n_nonterminals: int,
    n_lexical_rules: int,
    n_binary_rules: int,
    data_dir: pathlib.Path = PROJECT_ROOT / "data",
    name: str = "sample_raw",
    save_grammar: bool = True,
) -> dict[str, Any]:
    assert n_terminals > 0, "Number of terminals must be greater than 0"
    assert n_nonterminals > 0, "Number of nonterminals must be greater than 0"
    assert n_lexical_rules > 0, "Number of lexical rules must be greater than 0"
    assert n_binary_rules > 0, "Number of binary rules must be greater than 0"

    symbol_dict = get_symbols(n_terminals, n_nonterminals)
    terminals = symbol_dict["terminals"]
    nonterminals = symbol_dict["nonterminals"]

    lprods = set()
    bprods = set()

    if n_lexical_rules > n_terminals * n_nonterminals:
        log.warning(
            f"{n_lexical_rules} lexical rules requested, but only"
            f" {n_terminals * n_nonterminals} possible"
        )
        n_lexical_rules = n_terminals * n_nonterminals

    while len(lprods) < n_lexical_rules:
        a = random.choice(nonterminals[1:])
        b = random.choice(terminals)
        lprods.add(Production(lhs=a, rhs=(b,)))

    if n_binary_rules > n_nonterminals * (n_nonterminals - 1) * (n_nonterminals - 1):
        log.warning(
            f"{n_binary_rules} binary rules requested, but only "
            f"{n_nonterminals * (n_nonterminals - 1) * (n_nonterminals - 1)} possible"
        )
        n_binary_rules = n_nonterminals * (n_nonterminals - 1) * (n_nonterminals - 1)

    while len(bprods) < n_binary_rules:
        a = random.choice(nonterminals)
        b, c = random.choices(nonterminals[1:], k=2)
        bprods.add(Production(lhs=a, rhs=(b, c)))

    productions = list(lprods) + list(bprods)
    filename = f"{name}_{datetime.datetime.now().strftime('%Y%m%d%H%M%S')}.cfg"
    filepath = data_dir / filename

    sampled_grammar = load_grammar(productions, filepath, save_grammar=save_grammar)
    return sampled_grammar | {
        "productions": productions,
        "terminals": terminals,
        "nonterminals": nonterminals,
    }


def sample_reg_raw(
    n_terminals: int,
    n_nonterminals: int,
    n_lexical_rules: int,
    n_binary_rules: int,
    data_dir: pathlib.Path = PROJECT_ROOT / "data",
    name: str = "sample_raw",
    save_grammar: bool = True,
) -> dict[str, Any]:
    assert n_terminals > 0, "Number of terminals must be greater than 0"
    assert n_nonterminals > 0, "Number of nonterminals must be greater than 0"
    assert n_lexical_rules > 0, "Number of lexical rules must be greater than 0"
    assert n_binary_rules > 0, "Number of binary rules must be greater than 0"

    symbol_dict = get_symbols(n_terminals, n_nonterminals)
    terminals = symbol_dict["terminals"]
    nonterminals = symbol_dict["nonterminals"]

    lprods = set()
    bprods = set()

    if n_lexical_rules > n_terminals * n_nonterminals:
        log.warning(
            f"{n_lexical_rules} lexical rules requested, but only",
            f" {n_terminals * n_nonterminals} possible",
        )
        n_lexical_rules = n_terminals * n_nonterminals

    while len(lprods) < n_lexical_rules:
        a = random.choice(nonterminals[1:])
        b = random.choice(terminals)
        lprods.add(Production(lhs=a, rhs=(b,)))

    if n_binary_rules > n_nonterminals * (n_nonterminals - 1) * (n_nonterminals - 1):
        log.warning(
            f"{n_binary_rules} binary rules requested, but only ",
            f"{n_nonterminals * (n_nonterminals - 1) * (n_nonterminals - 1)} possible",
        )
        n_binary_rules = n_nonterminals * (n_nonterminals - 1) * (n_nonterminals - 1)

    while len(bprods) < n_binary_rules:
        a = random.choice(nonterminals)
        b = random.choice(nonterminals[1:])
        c = random.choice(terminals)

        print(a)
        print(b)
        print(c)

        bprods.add(Production(lhs=a, rhs=(b, c)))

    productions = list(lprods) + list(bprods)
    filename = f"{name}_{datetime.datetime.now().strftime('%Y%m%d%H%M%S')}.cfg"
    filepath = data_dir / filename

    sampled_grammar = load_grammar(productions, filepath, save_grammar=save_grammar)
    return sampled_grammar | {
        "productions": productions,
        "terminals": terminals,
        "nonterminals": nonterminals,
    }


def sample_cfg_trim(
    n_terminals: int,
    n_nonterminals: int,
    n_lexical_rules: int,
    n_binary_rules: int,
    data_dir: pathlib.Path,
    name: str = "grammar",
    save_raw_grammar: bool = False,
    max_tries: int = 100,
) -> dict[str, Any]:
    has_generated_nonempty_grammar = False

    while not has_generated_nonempty_grammar and max_tries > 0:
        raw_grammar = sample_cfg_raw(
            n_terminals=n_terminals,
            n_nonterminals=n_nonterminals,
            n_lexical_rules=n_lexical_rules,
            n_binary_rules=n_binary_rules,
            data_dir=data_dir,
            save_grammar=save_raw_grammar,
        )

        trim_set = compute_trim_set(
            productions=raw_grammar["productions"],
            nonterminals=raw_grammar["nonterminals"],
            terminals=raw_grammar["terminals"],
        )

        if len(trim_set) != 0:
            has_generated_nonempty_grammar = True
        else:
            log.warning("Empty language!")
            max_tries -= 1

        if max_tries == 0:
            log.error(
                "Max tries exceeded! Unable to generate grammar with "
                "provided hyperparameters."
            )
            return None

    prods = compute_usable_prods(
        trim_set=trim_set, productions=raw_grammar["productions"]
    )

    terminals = set()
    for prod in prods:
        if prod.is_lexical:
            terminals.add(prod.rhs[0])

    # get a random number to append to the grammar name
    rnd = random.randint(0, 1_000_000)

    grammar_name = f"{name}_{datetime.datetime.now().strftime('%Y%m%d%H%M%S')}_{rnd}"
    grammar_path = data_dir / grammar_name
    grammar_path.mkdir(parents=True, exist_ok=True)
    filename = f"{grammar_name}.cfg"
    filepath = grammar_path / filename

    sampled_grammar = load_grammar(
        productions=prods, filepath=filepath, save_grammar=True
    )
    return sampled_grammar | {
        "grammar_path": grammar_path,
        "grammar_name": grammar_name,
    }


def sample_reg_trim(
    n_terminals: int,
    n_nonterminals: int,
    n_lexical_rules: int,
    n_binary_rules: int,
    data_dir: pathlib.Path,
    name: str = "grammar",
    save_raw_grammar: bool = False,
    max_tries: int = 100,
) -> dict[str, Any] | None:
    has_generated_nonempty_grammar = False

    while not has_generated_nonempty_grammar and max_tries > 0:
        raw_grammar = sample_reg_raw(
            n_terminals=n_terminals,
            n_nonterminals=n_nonterminals,
            n_lexical_rules=n_lexical_rules,
            n_binary_rules=n_binary_rules,
            data_dir=data_dir,
            save_grammar=save_raw_grammar,
        )

        trim_set = compute_trim_set(
            productions=raw_grammar["productions"],
            nonterminals=raw_grammar["nonterminals"],
            terminals=raw_grammar["terminals"],
        )

        if len(trim_set) != 0:
            has_generated_nonempty_grammar = True
        else:
            log.warning("Empty language!")
            max_tries -= 1

        if max_tries == 0:
            log.error(
                "Max tries exceeded! Unable to generate grammar with "
                "provided hyperparameters."
            )
            return None

    prods = compute_usable_prods(
        trim_set=trim_set, productions=raw_grammar["productions"]
    )

    terminals = set()
    for prod in prods:
        if prod.is_lexical:
            terminals.add(prod.rhs[0])

    grammar_name = f"{name}_{datetime.datetime.now().strftime('%Y%m%d%H%M%S')}"
    grammar_path = data_dir / grammar_name
    grammar_path.mkdir(parents=True, exist_ok=True)
    filename = f"{grammar_name}.reg"
    filepath = grammar_path / filename

    sampled_grammar = load_grammar(
        productions=prods, filepath=filepath, save_grammar=True
    )
    return sampled_grammar | {
        "grammar_path": grammar_path,
        "grammar_name": grammar_name,
    }
