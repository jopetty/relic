"""Context-free grammar."""

import dataclasses

import numpy as np


@dataclasses.dataclass(frozen=True)
class CFGConfig:
    n_nonterminals: int
    n_terminals: int
    n_binary_rules: int
    n_lexical_rules: int
    strict_cnf: bool = True


class CFG:
    """A CFG in Chomsky normal form."""

    @property
    def n_nonterminals(self) -> int:
        return len(self.nonterminals)
    
    @property
    def n_terminals(self) -> int:
        return len(self.terminals)
    
    @property
    def n_binary_productions(self) -> int:
        return len([p for p in self.productions if len(p) == 3])
    
    @property
    def n_lexical_productions(self) -> int:
        return len([p for p in self.productions if len(p) == 2])
    
    @property
    def n_productions(self) -> int:
        return len(self.productions)

    def __init__(self):
        self.start = None
        self.nonterminals = set()
        self.productions = set()
        self.terminals = set()
    
    # Class methods

    @classmethod
    def sample_full(cls, config: CFGConfig):
        terminals = [f'w{i}' for i in range(config.n_terminals)]
        nonterminals = ['S'] + [f'N{i}' for i in range(config.n_nonterminals)]
        lprods = set()
        bprods = set()

        for a, w in zip(nonterminals, terminals):
            lprods.add((a, w))
        
        for a, b, c in zip(nonterminals, nonterminals[1:], nonterminals[1:]):
            bprods.add((a, b, c))
        
        cfg = cls()
        cfg.start = 'S'
        cfg.nonterminals = set(nonterminals)
        cfg.terminals = set(terminals)
        cfg.productions = lprods | bprods
        return cfg

    @classmethod
    def sample_uniform(cls, config: CFGConfig, lp = 0.5, bp = 0.5):
        terminals = [f'w{i}' for i in range(config.n_terminals)]
        nonterminals = ['S'] + [f'N{i}' for i in range(config.n_nonterminals)]
        productions = set()

        for a, w in zip(nonterminals, terminals):
            if np.random.random() < lp:
                productions.add((a, w))
        
        for a, b, c in zip(nonterminals, nonterminals[1:], nonterminals[1:]):
            if np.random.random() < bp:
                productions.add((a, b, c))
        
        cfg = cls()
        cfg.start = 'S'
        cfg.nonterminals = set(nonterminals)
        cfg.terminals = set(terminals)
        cfg.productions = productions
        return cfg


if __name__ == '__main__':

    config = CFGConfig(
        n_nonterminals=100,
        n_terminals=100,
        n_binary_rules=100,
        n_lexical_rules=100
    )

    cfg_a = CFG.sample_full(config)
    print(cfg_a.n_nonterminals)
    print(cfg_a.n_terminals)
    print(cfg_a.n_productions)
    print(cfg_a.productions)

    cfg_b = CFG.sample_uniform(config)
    print(cfg_b.productions)
