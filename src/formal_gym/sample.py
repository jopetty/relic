"""Sample from grammars and metagrammars."""

import logging
import pathlib

import fire
import pyrootutils

import formal_gym.grammar as fg_grammar
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


def sample_expression(
    grammar_file: pathlib.Path | str = "data/untyped_lambda_calculus.cfg",
    n_samples: int = 5,
):
    if isinstance(grammar_file, str):
        grammar_file = pathlib.Path(grammar_file)

    log.info(f"Generating {n_samples} samples from {grammar_file}")

    g = fg_grammar.Grammar.from_grammar(grammar_file)
    generations = list(g.generate(n_samples=n_samples, sep=" "))
    for i, gen in enumerate(generations):
        print(f"{i+1}: {gen}")


def sample_positive_examples(
    grammar_file: pathlib.Path | str = "data/untyped_lambda_calculus.cfg",
    n_samples: int = 5,
):
    if isinstance(grammar_file, str):
        grammar_file = pathlib.Path(grammar_file)

    # get filename without extension
    grammar_file_name = grammar_file.stem
    grammar_file_parent = grammar_file.parent
    outfile_name = f"{grammar_file_name}_positive_{n_samples}.txt"
    outfile_path = grammar_file_parent / outfile_name

    log.info(f"Generating {n_samples} positive examples from {grammar_file}")

    g = fg_grammar.Grammar.from_grammar(grammar_file)
    generations = list(g.generate(n_samples=n_samples, sep=" "))

    log.info(f"Writing to {outfile_path}")
    with open(outfile_path, "w") as f:
        for gen in generations:
            f.write(f"{gen}\n")


def sample_negative_examples(
    grammar_file: pathlib.Path | str = "data/untyped_lambda_calculus.cfg",
    n_samples: int = 5,
    s_max_length: int = 20,
):
    if isinstance(grammar_file, str):
        grammar_file = pathlib.Path(grammar_file)

    # get filename without extension
    grammar_file_name = grammar_file.stem
    grammar_file_parent = grammar_file.parent
    outfile_name = f"{grammar_file_name}_negative_{n_samples}.txt"
    outfile_path = grammar_file_parent / outfile_name

    log.info(f"Generating {n_samples} negative examples from {grammar_file}")

    g = fg_grammar.Grammar.from_grammar(grammar_file)
    generations = list(
        g.generate_negative_samples(
            n_samples=n_samples,
            s_max_length=s_max_length,
        )
    )

    log.info(f"Writing to {outfile_path}")
    with open(outfile_path, "w") as f:
        for gen in generations:
            f.write(f"{gen}\n")


def sample_negative_examples_matching(
    positive_sample_path: pathlib.Path | str,
):
    if isinstance(positive_sample_path, str):
        positive_sample_path = pathlib.Path(positive_sample_path)

    with open(positive_sample_path, "r") as f:
        positive_samples = f.readlines()
    n_samples = len(positive_samples)

    # get filename without extension
    grammar_file = positive_sample_path.stem.split("_positive")[0]
    grammar_file = pathlib.Path("data") / f"{grammar_file}.cfg"
    grammar_file_name = grammar_file.stem
    grammar_file_parent = grammar_file.parent
    outfile_name = f"{grammar_file_name}_negative_{n_samples}.txt"
    outfile_path = grammar_file_parent / outfile_name

    log.info(
        f"Generating {n_samples} negative examples from {grammar_file} ",
        f"matching lengths of {positive_sample_path}",
    )

    g = fg_grammar.Grammar.from_grammar(grammar_file)
    generations = list(
        g.generate_negative_samples_matching_lengths(
            positive_sample_path=positive_sample_path,
        )
    )

    log.info(f"Writing to {outfile_path}")
    with open(outfile_path, "w") as f:
        for gen in generations:
            f.write(f"{gen}\n")


def sample_grammar():
    # g_uniform = fg_metagrammar.sample_cfg_uniform(
    #     n_terminals=5,
    #     n_nonterminals=3,
    # )["grammar"]

    # print(f"Uniform sampled:\n{g_uniform.grammar_obj}")

    # g_full = fg_metagrammar.sample_cfg_full(
    #     n_terminals=5,
    #     n_nonterminals=3,
    # )["grammar"]

    # print(f"Full sampled:\n{g_full.grammar_obj}")

    # g_raw = fg_metagrammar.sample_cfg_raw(
    #     n_terminals=5,
    #     n_nonterminals=3,
    #     n_binary_rules=5,
    #     n_lexical_rules=5,
    # )["grammar"]

    # print(f"Raw sampled:\n{g_raw.grammar_obj}")

    g_trim = fg_metagrammar.sample_cfg_trim(
        n_terminals=5,
        n_nonterminals=6,
        n_binary_rules=6,
        n_lexical_rules=5,
    )["grammar"]

    print(f"Trim sampled:\n{g_trim.grammar_obj}")


if __name__ == "__main__":
    fire.Fire()
