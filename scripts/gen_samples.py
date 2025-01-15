import logging
import pathlib

import fire
import pyrootutils
import tqdm

from formal_gym import grammar as fg_grammar
from formal_gym.utils import utils as fg_utils

PROJECT_ROOT = path = pyrootutils.find_root(
    search_from=__file__, indicator=".project-root"
)

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%Y-%d-%m %H:%M:%S",
    level=logging.INFO,
)

log = fg_utils.get_logger(__name__)


def main(
    grammar_file: pathlib.Path | str,
    data_dir: pathlib.Path | str = PROJECT_ROOT / "data",
    max_samples: int = 10_000,
    max_recursion_depth: int = 100,
    max_neg_trials: int = 10,
    max_neg_length: int = 100,
    negative: bool = False,
):
    if isinstance(grammar_file, str):
        grammar_file = pathlib.Path(grammar_file)

    if isinstance(data_dir, str):
        data_dir = pathlib.Path(data_dir)

    grammar_path = data_dir / grammar_file

    samples_dir = data_dir / "samples"
    grammar_samples_dir = samples_dir / grammar_file.stem
    grammar_samples_dir.mkdir(parents=True, exist_ok=True)

    outpath = grammar_samples_dir / f"{'negative' if negative else 'positive'}.txt"

    samples = set()
    # check if outpath exists; if so, load the samples and update the samples set
    if outpath.exists():
        with open(outpath, "r") as f:
            samples.update(f.read().splitlines())

    starting_len = len(samples)

    log.info(
        f"Generating {'negative' if negative else 'positive'} samples for {grammar_path}"
    )
    grammar = fg_grammar.ContextFreeGrammar.from_file(grammar_path)
    for _ in tqdm.tqdm(range(max_samples)):
        if negative:
            sample = grammar.generate_negative_sample(
                max_trials=max_neg_trials, max_length=max_neg_length
            )
        else:
            sample = grammar.generate(max_depth=max_recursion_depth)

        if sample is not None:
            samples.add(sample)

    ending_len = len(samples)

    log.info(f"Generated {ending_len - starting_len} samples")
    log.info(f"Total samples: {ending_len}")
    log.info(f"Writing samples to {outpath}")
    with open(outpath, "w") as f:
        for sample in samples:
            f.write(f"{sample}\n")


if __name__ == "__main__":
    fire.Fire(main)
