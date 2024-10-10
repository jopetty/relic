import logging
from pathlib import Path

import fire
import pyrootutils

import formal_gym.grammar as fg_grammar
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


def sample_expression(
    grammar_file: Path | str = "data/untyped_lambda_calculus.cfg",
    n_samples: int = 5,
):
    if isinstance(grammar_file, str):
        grammar_file = Path(grammar_file)

    log.info(f"Generating {n_samples} samples from {grammar_file}")

    g = fg_grammar.Grammar.from_grammar(grammar_file)
    generations = list(g.generate(n_samples=n_samples))
    for i, gen in enumerate(generations):
        print(f"{i+1}: {gen}")


def sample_grammar():
    fg_grammar.Grammar.sample_cfg(
        n_terminals=5,
        n_nonterminals=5,
        # n_binary_rules=5,
        # n_lexical_rules=5,
    )


if __name__ == "__main__":
    fire.Fire()
