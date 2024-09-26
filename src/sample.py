import logging
from pathlib import Path

import fire
import grammar
import pyrootutils
from utils.utils import get_logger

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%Y-%d-%m %H:%M:%S",
    level=logging.INFO,
)

log = get_logger(__name__)

PROJECT_ROOT = path = pyrootutils.find_root(
    search_from=__file__, indicator=".project-root"
)


def sample_expression(
    grammar_file: Path | str = Path("data/untyped_lambda_calculus.cfg"),
    n_samples: int = 5,
):
    if isinstance(grammar_file, str):
        grammar_file = Path(grammar_file)

    g = grammar.Grammar.from_grammar(grammar_file)
    generations = list(g.generate(n_samples=n_samples))
    for i, gen in enumerate(generations):
        print(f"{i+1}: {gen}")


if __name__ == "__main__":
    fire.Fire(sample_expression)
