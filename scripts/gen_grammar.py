import json
import logging
import pprint
from pathlib import Path

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
    grammar_dict = fg_metagrammar.sample_cfg_trim(
        n_terminals=n_terminals,
        n_nonterminals=n_nonterminals,
        n_binary_rules=n_binary_rules,
        n_lexical_rules=n_lexical_rules,
        data_dir=data_dir,
    )
    g = grammar_dict["grammar"]

    grammar_stats = {
        "n_terminals": g.n_terminals,
        "n_nonterminals": g.n_nonterminals,
        "n_lexical_productions": g.n_lexical_productions,
        "n_nonlexical_productions": g.n_nonlexical_productions,
        "grammar_name": grammar_dict["grammar_name"],
    }

    grammar_stats_file = grammar_dict["grammar_path"] / "grammar_stats.json"
    with open(grammar_stats_file, "w") as f:
        json.dump(grammar_stats, f, indent=4)

    print("Grammar:")
    print(g.grammar_obj)

    pprint.pprint(grammar_stats)


if __name__ == "__main__":
    fire.Fire(grammar)
