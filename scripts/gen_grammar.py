import logging

import fire
import pyrootutils

import formal_gym.metagrammar as fg_metagrammar
import formal_gym.utils.utils as fg_utils

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%Y-%d-%m %H:%M:%S",
    level=logging.INFO,
)

log = fg_utils.get_logger(__name__)

PROJECT_ROOT = path = pyrootutils.find_root(
    search_from=__file__, indicator=".project-root"
)


def grammar(
    n_terminals=5,
    n_nonterminals=6,
    n_binary_rules=6,
    n_lexical_rules=5,
    data_dir=PROJECT_ROOT / "data" / "grammars",
):
    g = fg_metagrammar.sample_cfg_trim(
        n_terminals=n_terminals,
        n_nonterminals=n_nonterminals,
        n_binary_rules=n_binary_rules,
        n_lexical_rules=n_lexical_rules,
        data_dir=data_dir,
    )["grammar"]

    print(
        f"Sampled grammar [n_term: {g.n_terminals}, n_nonterm: {g.n_nonterminals}, n_lex_rules:{g.n_lexical_productions}, n_nonlex_rules: {g.n_non_lexical_productions}]\n{g.grammar_obj}"
    )


if __name__ == "__main__":
    fire.Fire(grammar)
