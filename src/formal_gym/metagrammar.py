"""Sample from grammar space."""

import collections
import datetime
import pathlib
from typing import Any, NamedTuple, Sequence

import numpy as np
import pyrootutils

import formal_gym.grammar as fg_grammar
import formal_gym.utils.utils as fg_utils


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
            if all([symbol in trim_set for symbol in prod.rhs]):
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

    if "S" in coreachable:
        trim.add("S")
    done = len(trim)
    while done > 0:
        done = 0
        for prod in good_productions:
            a, rhs = prod.split(" -> ")
            rhs = rhs.split(" ")
            if a in trim:
                for symbol in rhs:
                    if (symbol in nonterminals) and (symbol not in trim):
                        done += 1
                        trim.add(symbol)
    print(f"Coreachable: {coreachable}")
    print(f"Trim: {trim}")
    print(f"Good productions: {good_productions}")
    return trim


def prod_to_string(prod: Production) -> str:
    rhs_string = " ".join(
        [s if isinstance(s, Nonterminal) else f"'{s}'" for s in prod.rhs]
    )
    return f"{prod.lhs} -> {rhs_string}"


def save_and_load_grammar(
    productions: Sequence[Production],
    filepath: pathlib.Path,
) -> dict[str, Any]:
    """
    Save a grammar to a file and load it back.

    Args:
        productions: List of productions.
        filepath: Path to save the grammar.
    """

    grammar_string = "\n".join([prod_to_string(p) for p in productions])
    with open(filepath, "w") as f:
        f.write(grammar_string)

    log.info(f"Grammar saved to {filepath}")

    return {
        "filepath": filepath,
        "grammar": fg_grammar.Grammar.from_grammar(filepath),
    }


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
    terminals = [Terminal(f"t{i}") for i in range(n_terminals)]
    nonterminals = [Nonterminal("S")] + [
        Nonterminal(f"NT{i}") for i in range(n_nonterminals)
    ]
    productions = []
    for a in nonterminals:
        for b in terminals:
            if np.random.random() < lp:
                productions.append(Production(lhs=a, rhs=[b]))
    for a in nonterminals:
        for b in nonterminals[1:]:
            for c in nonterminals[1:]:
                if np.random.random() < bp:
                    productions.append(Production(lhs=a, rhs=[b, c]))
    filename = f"{name}_{datetime.datetime.now().strftime('%Y%m%d%H%M%S')}.cfg"
    filepath = data_dir / f"{filename}.cfg"
    sampled_grammar = save_and_load_grammar(productions, filepath)
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
    terminals = [f"t{i}" for i in range(n_terminals)]
    nonterminals = ["S"] + [f"NT{i}" for i in range(n_nonterminals)]
    productions = []
    for a in nonterminals:
        for b in terminals:
            productions.append(f"{a} -> '{b}'")
    for a in nonterminals:
        for b in nonterminals[1:]:
            for c in nonterminals[1:]:
                productions.append(f"{a} -> {b} {c}")
    filename = f"{name}_{datetime.datetime.now().strftime('%Y%m%d%H%M%S')}.cfg"
    filepath = data_dir / f"{filename}.cfg"
    sampled_grammar = save_and_load_grammar(productions, filepath)
    return sampled_grammar


def sample_cfg_raw(
    n_terminals: int,
    n_nonterminals: int,
    n_lexical_rules: int,
    n_binary_rules: int,
    data_dir: pathlib.Path = PROJECT_ROOT / "data",
    name: str = "sample_raw",
) -> dict[str, Any]:
    terminals = [Terminal(f"t{i}") for i in range(n_terminals)]
    nonterminals = [START_SYMBOL] + [
        Nonterminal(f"NT{i}") for i in range(n_nonterminals)
    ]

    lprods = set()
    bprods = set()

    while len(lprods) < n_lexical_rules:
        a = np.random.choice(nonterminals)
        b = np.random.choice(terminals)
        lprods.add(Production(lhs=a, rhs=(b)))

    while len(bprods) < n_binary_rules:
        a = np.random.choice(nonterminals)
        b, c = np.random.choice(nonterminals[1:], size=2)
        bprods.add(Production(lhs=a, rhs=(b, c)))

    productions = list(lprods) + list(bprods)
    filename = f"{name}_{datetime.datetime.now().strftime('%Y%m%d%H%M%S')}.cfg"
    filepath = data_dir / f"{filename}"

    sampled_grammar = save_and_load_grammar(productions, filepath)
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
    data_dir: pathlib.Path = PROJECT_ROOT / "data",
    name: str = "sample_trim",
) -> dict[str, Any]:
    raw_grammar = sample_cfg_raw(
        n_terminals=n_terminals,
        n_nonterminals=n_nonterminals,
        n_lexical_rules=n_lexical_rules,
        n_binary_rules=n_binary_rules,
        data_dir=data_dir,
    )

    trim_set = compute_trim_set(
        productions=raw_grammar["productions"],
        nonterminals=raw_grammar["nonterminals"],
        terminals=raw_grammar["terminals"],
    )

    if len(trim_set) == 0:
        log.error("Empty language!")
        raise ValueError("Empty language!")

    prods = compute_usable_prods(
        trim_set=trim_set, productions=raw_grammar["productions"]
    )

    terminals = set()
    for prod in prods:
        _, rhs = prod.split(" -> ")
        rhs = rhs.split(" ")
        if len(rhs) == 1:
            terminals.add(rhs[0])

    filename = f"{name}_{datetime.datetime.now().strftime('%Y%m%d%H%M%S')}.cfg"
    filepath = data_dir / f"{filename}.cfg"

    sampled_grammar = save_and_load_grammar(productions=prods, filepath=filepath)
    return sampled_grammar


if __name__ == "__main__":
    terminal1 = Terminal("t1")
    terminal2 = Terminal("t2")
    nonterminal = Nonterminal("NT")

    prod1 = Production(lhs=nonterminal, rhs=(terminal1, terminal2))

    print(terminal1)
    print(terminal2)
    print(nonterminal)
    print(prod1)
    print(prod1.length)
