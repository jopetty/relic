# generate.py
#
# Generates data for grammar evaluation, including:
#   - sampling grammars
#   - generating positive and negative samples from a grammar
#   - filtering samples by length
#   - generating batch job files for LLM APIs
#   - calculating statistics on grammars and their samples

import gzip
import itertools
import json
import logging
import pprint
import random
import shutil
from dataclasses import asdict
from typing import Any

import fire
import pandas as pd
import pyrootutils
import tqdm

import formal_gym.grammar as fg_grammar
import formal_gym.metagrammar as fg_metagrammar
import formal_gym.metaxbargrammar as fg_mxg
import formal_gym.prompt as fg_prompt
import formal_gym.utils.utils as fg_utils

GType = fg_grammar.Grammar.Type
GrammarParams = fg_mxg.GrammarParams

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%Y-%d-%m %H:%M:%S",
    level=logging.INFO,
)

log = fg_utils.get_logger(__name__)

PROJECT_ROOT = path = pyrootutils.find_root(
    search_from=__file__, indicator=".project-root"
)

# Define some starting constants for grammar hyperparams.
N_TERMINALS = 1000
N_NONTERMINALS = 2000
N_LEXICAL_RULES = 2000
N_NONLEXICAL_RULES = 2000


def grammar(
    n_terminals=N_TERMINALS,
    n_nonterminals=N_NONTERMINALS,
    n_lexical_rules=N_LEXICAL_RULES,
    n_nonlexical_rules=N_NONLEXICAL_RULES,
    type: str = "cfg",
):
    grammars_dir = PROJECT_ROOT / "data" / "grammars"

    if type == "cfg":
        grammar_dict = fg_metagrammar.sample_cfg_trim(
            n_terminals=n_terminals,
            n_nonterminals=n_nonterminals,
            n_lexical_rules=n_lexical_rules,
            n_binary_rules=n_nonlexical_rules,
            data_dir=grammars_dir,
        )
    elif type == "reg":
        grammar_dict = fg_metagrammar.sample_reg_trim(
            n_terminals=n_terminals,
            n_nonterminals=n_nonterminals,
            n_lexical_rules=n_lexical_rules,
            n_binary_rules=n_nonlexical_rules,
            data_dir=grammars_dir,
        )
    else:
        raise ValueError(
            f"Unknown grammar type: `{type}`; valid types are `cfg` and `reg`"
        )

    if grammar_dict is None:
        log.warning("Failed to generate grammar")
        return None

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
    print(g.as_cfg)

    pprint.pprint(grammar_stats)

    return grammar_dict


def grammars():
    n_terminals_range = range(100, 500, 50)
    n_nonterminals_range = range(100, 500, 50)
    n_lexical_rules_range = range(100, 500, 50)
    n_nonlexical_rules_range = range(100, 500, 50)

    new_grammars = []
    for (
        n_terminals,
        n_nonterminals,
        n_lexical_rules,
        n_nonlexical_rules,
    ) in tqdm.tqdm(
        itertools.product(
            n_terminals_range,
            n_nonterminals_range,
            n_lexical_rules_range,
            n_nonlexical_rules_range,
        )
    ):
        try:
            grammar_dict = grammar(
                n_terminals=n_terminals,
                n_nonterminals=n_nonterminals,
                n_lexical_rules=n_lexical_rules,
                n_nonlexical_rules=n_nonlexical_rules,
            )
            if grammar_dict is not None:
                new_grammars.append(grammar_dict)
        except Exception as _:
            log.warning(
                f"Unable to generate grammar with params: {n_terminals=}, {n_nonterminals=}, {n_lexical_rules=}, {n_nonlexical_rules=}"
            )

    log.info(f"Generated {len(new_grammars)} new grammars")

    for g_dict in new_grammars:
        log.info(f"\tGrammar: {g_dict['grammar_name']}")

    for g_dict in new_grammars:
        # generate samples for each grammar
        samples(grammar_name=g_dict["grammar_name"])
        filtered_samples(grammar_name=g_dict["grammar_name"])


def samples(
    grammar_name: str,
    max_length: int = 50,
    samples_per_length: int = 10,
    gen_positive: bool = True,
    gen_negative: bool = True,
    max_tries_per_sample: int = 10,
    max_recursion_depth: int = 100,
    pos_multiplier: int = 100,
):
    grammars_dir = PROJECT_ROOT / "data" / "grammars"

    grammar_path = grammars_dir / f"{grammar_name}"
    grammar_file = grammar_path / f"{grammar_name}.cfg"
    if not grammar_path.exists():
        raise FileNotFoundError(f"Grammar director `{grammar_name}` not found")

    g = fg_grammar.Grammar.from_file(grammar_file)

    if gen_negative:
        neg_sample_file = grammar_path / "negative_samples.txt"
        if neg_sample_file.exists():
            with open(neg_sample_file, "r") as f:
                neg_samples = set(f.read().splitlines())
        else:
            neg_samples = set()
        starting_count = len(neg_samples)

        # Rejection sampling
        # Generate twenty random strings of varying lengths between 1 and
        # max_length; if none of them are negative samples, then the grammar
        # probably overgenerates and isn't good.

        terminals: list[str] = list(set(g.terminals))

        log.info("Performing rejection sampling check...")
        found_negative_in_rejection = False
        for _ in range(20):  # Generate 20 random strings
            length = random.randint(1, max_length)
            random_terminals = random.choices(terminals, k=length)
            sample = " ".join(random_terminals)
            if not g.test_sample(sample):
                neg_samples.add(sample)
                found_negative_in_rejection = True
                # Optionally break early if one is found, or continue to find more
                # break

        if not found_negative_in_rejection:
            log.warning(
                f"Grammar {grammar_name} might overgenerate: "
                "No negative samples found in 20 random strings."
            )
            raise SystemExit

        log.info(f"Generating negative samples for {grammar_file}")
        n_terminals: int = g.n_terminals
        test_sample_len: int = 1

        while (n_terminals**test_sample_len < 5_000) and (test_sample_len < 50):
            log.info(f"Testing short strings of length {test_sample_len}")
            for possible_sample in itertools.product(terminals, repeat=test_sample_len):
                sample = " ".join(possible_sample)
                if not g.test_sample(sample):
                    neg_samples.add(sample)
            test_sample_len += 1

        log.info("Generating longer samples")
        for length in tqdm.tqdm(range(test_sample_len - 1, max_length + 1)):
            bad_tries = 0
            for _ in range(samples_per_length):
                sample = g.generate_negative_sample_of_length(
                    length=length, max_trials=max_tries_per_sample
                )
                if sample is not None:
                    neg_samples.add(sample)
                else:
                    bad_tries += 1
                    if bad_tries > max_tries_per_sample:
                        print(f"failed {bad_tries} times")
                        break
        ending_count = len(neg_samples)
        log.info(
            f"Generated {ending_count - starting_count} new negative samples"
            f" ({len(neg_samples)} total)"
        )

        # sort neg_samples by length
        neg_samples = sorted(neg_samples, key=lambda x: len(x.split(" ")))

        with open(neg_sample_file, "w") as f:
            for sample in neg_samples:
                f.write(f"{sample}\n")

    if gen_positive:
        pos_sample_file = grammar_path / "positive_samples.csv"

        if pos_sample_file.exists():
            pos_samples_df = pd.read_csv(pos_sample_file)
        else:
            pos_samples_df = pd.DataFrame(columns=["string", "parse"])
        starting_count = pos_samples_df["string"].nunique()
        log.info(f"Generating positive samples for {grammar_file}")

        total_iterations = (
            samples_per_length * max_length * max_tries_per_sample * pos_multiplier
        )

        new_samples = []
        for _ in tqdm.tqdm(range(total_iterations)):
            sample = g.generate_tree(max_depth=max_recursion_depth)
            if sample is not None:
                new_samples.append(sample)

        # add new_samples to pos_samples_df
        new_samples_df = pd.DataFrame(new_samples, columns=["string", "parse"])
        pos_samples_df = pd.concat([pos_samples_df, new_samples_df])

        ending_count = pos_samples_df["string"].nunique()

        pos_samples_df = pos_samples_df.drop_duplicates(subset=["string"])
        log.info(
            f"Generated {ending_count - starting_count} new positive samples"
            f" ({ending_count} total)"
        )

        pos_samples_df.sort_values(by="string", key=lambda x: x.str.len()).to_csv(
            pos_sample_file, index=False
        )


def filtered_samples(
    grammar_name: str,
    max_length: int = 50,
    samples_per_length: int = 10,
):
    def read_lines_up_to_length(filename, length):
        with open(filename, "r") as f:
            lines = []
            for line in f:
                line = line.strip()
                if len(line.split(" ")) > length:
                    break
                lines.append(line)
        return lines

    grammars_dir = PROJECT_ROOT / "data" / "grammars"

    grammar_path = grammars_dir / f"{grammar_name}"
    grammar_file = grammar_path / f"{grammar_name}.cfg"
    if not grammar_path.exists():
        raise FileNotFoundError(f"Grammar director `{grammar_name}` not found")

    g = fg_grammar.Grammar.from_file(grammar_file)

    # assert that positive and negative samples have been generated
    pos_sample_file = grammar_path / "positive_samples.csv"
    neg_sample_file = grammar_path / "negative_samples.txt"
    if not pos_sample_file.exists():
        raise FileNotFoundError(f"Positive samples not found for {grammar_name}")
    if not neg_sample_file.exists():
        raise FileNotFoundError(f"Negative samples not found for {grammar_name}")

    # Read in only the first samples_per_length*max_length lines for each
    # sample file
    # pos_samples = read_lines_up_to_length(pos_sample_file, max_length)
    neg_samples = read_lines_up_to_length(neg_sample_file, max_length)

    # pos_samples = [s for s in pos_samples if len(s.split(" ")) <= max_length]
    neg_samples = [s for s in neg_samples if len(s.split(" ")) <= max_length]

    # Keep only 2*samples_per_length samples of each length to speed up parsing
    pos_samples_df = pd.read_csv(pos_sample_file)
    pos_samples_df = pos_samples_df.rename(columns={"string": "sample"})
    pos_samples_df["length"] = pos_samples_df["sample"].apply(
        lambda x: len(x.split(" "))
    )
    pos_samples_df = pos_samples_df[pos_samples_df["length"] <= max_length]

    pos_samples_df = pos_samples_df.groupby(["length"], group_keys=False).apply(
        lambda x: x.sample(min(len(x), 2 * samples_per_length)),
        include_groups=False,
    )
    pos_samples = pos_samples_df["sample"].tolist()

    log.info(f"Read in {len(pos_samples)} positive samples")
    log.info(f"Read in {len(neg_samples)} negative samples")

    def get_dyck_depth(s):
        depth = 0
        max_depth = 0

        for c in s:
            if c == "(":
                depth += 1
                max_depth = max(max_depth, depth)
            elif c == ")":
                depth -= 1

        return max_depth

    pos_samples_df["parse_depth"] = pos_samples_df["parse"].apply(get_dyck_depth)

    # def annotate_sample(s):
    #     parses = g.parse(s)
    #     num_parses = len(parses)
    #     mean_parse_depth = statistics.mean(p.height() for p in parses)
    #     return {
    #         "sample": s,
    #         "mean_parse_height": mean_parse_depth,
    #         "num_parses": num_parses,
    #         "type": "positive",
    #         "length": len(s.split(" ")),
    #     }

    # log.info("Annotating positive samples with parse information")
    # with ThreadPoolExecutor(max_workers=8) as executor:
    #     futures = {executor.submit(annotate_sample, s): s for s in pos_samples}
    #     sample_dicts = []
    #     for future in tqdm.tqdm(as_completed(futures), total=len(pos_samples)):
    #         sample_dicts.append(future.result())

    sample_dicts = []

    # convert pos_samples_df to list of dicts
    for idx, row in pos_samples_df.iterrows():
        sample_dicts.append(
            {
                "sample": row["sample"],
                "parse_depth": get_dyck_depth(row["parse"]),
                "type": "positive",
                "length": len(row["sample"].split(" ")),
            }
        )

    for s in neg_samples:
        sample_dicts.append(
            {
                "sample": s,
                "parse_depth": None,
                # "parse_depth": get_dyck_depth(s),
                # "mean_parse_height": None,
                # "num_parses": None,
                "type": "negative",
                "length": len(s.split(" ")),
            }
        )

    sample_df = pd.DataFrame(sample_dicts)

    sample_df["type"] = pd.Categorical(
        sample_df["type"], categories=["positive", "negative"]
    )
    sample_df["length_cat"] = pd.Categorical(sample_df["length"])

    # group samples by length, select `samples_per_length` many per length
    sample_df = sample_df.groupby(
        ["length", "type"], group_keys=False, observed=True
    ).apply(
        lambda x: x.sample(min(len(x), samples_per_length)),
        include_groups=True,
    )

    # First, we exclude any lengths which have the maximum number of samples
    n_terminals = g.n_terminals
    counts_df = (
        sample_df.groupby(["length"], group_keys=False)["sample"].count().to_frame()
    )

    total_samples = int(counts_df["sample"].sum())
    total_possible_samples = 2 * samples_per_length * max_length
    coverage = total_samples / float(total_possible_samples)

    counts_df["num_strings_of_length"] = (
        n_terminals ** counts_df.index.get_level_values("length")
    )
    counts_df["maxed_out"] = counts_df["sample"] == counts_df["num_strings_of_length"]

    maxed_out_lengths: list[int] = (
        counts_df[counts_df["maxed_out"]].index.get_level_values("length").tolist()
    )

    counts_df = (
        sample_df.groupby(["length", "type"], group_keys=False, observed=True)["sample"]
        .count()
        .to_frame()
        .reset_index()
    )
    counts_df = counts_df[counts_df["sample"] < samples_per_length]
    counts_df = counts_df[~counts_df["length"].isin(maxed_out_lengths)]

    if len(counts_df) > 0:
        log.info("The following lengths/types need more samples:")
        counts_df["missing_samples"] = samples_per_length - counts_df["sample"]
        counts_df["%"] = counts_df["missing_samples"] / samples_per_length * 100
        counts_df = counts_df.reset_index(drop=True)
        counts_df = counts_df.drop(columns=["sample"])
        print(counts_df)

    # the `length_cat` column is only needed for the missing samples report,
    # and is duplicated from the `length` column, so we drop it
    sample_df_no_cat = sample_df.drop(columns=["length_cat"])
    sample_df_no_cat.to_csv(grammar_path / "filtered_samples.csv", index=False)

    # write positive samples to file
    uncompressed_path = grammar_path / "filtered_positive_samples.txt"
    compressed_path = grammar_path / "filtered_positive_samples.txt.gz"
    pos_samples = sample_df[sample_df["type"] == "positive"]["sample"]
    with open(uncompressed_path, "w") as f:
        for s in pos_samples:
            f.write(f"{s}\n")

    # compress the positive samples using gzip
    with open(uncompressed_path, "rb") as f_in:
        with gzip.open(compressed_path, "wb") as f_out:
            shutil.copyfileobj(f_in, f_out)

    # calculate the compression ratio
    uncompressed_size = uncompressed_path.stat().st_size
    compressed_size = compressed_path.stat().st_size
    compression_ratio = uncompressed_size / compressed_size
    pos_depths = sample_df[sample_df["type"] == "positive"]["parse_depth"]

    samples_stats = {
        "uncompressed_size": uncompressed_size,
        "compressed_size": compressed_size,
        "compression_ratio": compression_ratio,
        "mean_positive_depth": pos_depths.mean().item(),
        "median_positive_depth": pos_depths.median().item(),
        "coverage": coverage,
        "total_samples": total_samples,
        "total_possible_samples": total_possible_samples,
    }

    samples_stats_file = grammar_path / "filtered_samples_stats.json"
    with open(samples_stats_file, "w") as f:
        json.dump(samples_stats, f, indent=4)

    pprint.pprint(samples_stats)


def openai_batch(
    grammar_name: str,
    model: str = "gpt-4o-mini",
    n_shots: int = 0,
    subsample_n: int | None = None,
):
    assert n_shots >= 0

    grammars_dir = PROJECT_ROOT / "data" / "grammars"
    grammar_path = grammars_dir / f"{grammar_name}"
    grammar_file = grammar_path / f"{grammar_name}.cfg"
    if not grammar_path.exists():
        raise FileNotFoundError(f"Grammar director `{grammar_name}` not found")

    grammar_str: str
    with open(grammar_file, "r") as f:
        grammar_str = f.read()

    samples_file = grammar_path / "filtered_samples.csv"
    samples_df = pd.read_csv(samples_file)

    pos_samples = samples_df[samples_df["type"] == "positive"].reset_index(drop=True)
    neg_samples = samples_df[samples_df["type"] == "negative"].reset_index(drop=True)

    if subsample_n is not None:
        # group samples_df by type and length, and keep only the first
        # `subsample_n` samples
        samples_df = samples_df.groupby(
            ["type", "length"], group_keys=False, observed=True
        ).apply(
            lambda x: x.sample(min(len(x), subsample_n)),
            include_groups=True,
        )

    samples_df["prompt"] = ""

    for idx, row in tqdm.tqdm(samples_df.iterrows(), total=len(samples_df)):
        if row["type"] == "positive":
            id_type = "positive"
            od_type = "negative"
            id_samples = pos_samples
            od_samples = neg_samples
        else:
            id_type = "negative"
            od_type = "positive"
            id_samples = neg_samples
            od_samples = pos_samples

        in_domain_idxs = [
            i
            for i in id_samples.index.values
            if id_samples.iloc[i]["sample"] != row["sample"]
        ]
        max_ood = max(od_samples.index.values)
        possible_idxs = [i for i in in_domain_idxs if i < max_ood]

        alt_ilocs = random.choices(possible_idxs, k=n_shots)
        id_alts = []
        od_alts = []
        for i in alt_ilocs:
            id_alts.append(id_samples.iloc[i]["sample"])
            od_alts.append(od_samples.iloc[i]["sample"])
        alt_samples = {
            id_type: id_alts,
            od_type: od_alts,
        }

        samples_df.at[idx, "prompt"] = fg_prompt.basic_prompt(
            grammar_str=grammar_str,
            sample=row["sample"],
            shots=alt_samples,
        )

    samples_df[f"{model}_batched_json"] = samples_df.apply(
        lambda row: fg_prompt.ChatCompletionResponse(
            user_prompt=row["prompt"],
            metadata={
                "sample_type": row["type"],
                "sample": row["sample"],
                "grammar_file": grammar_name,
                "model": model,
                "n_shots": str(2 * n_shots),
            },
        ).to_openai_batched_json(model=model, custom_id=f"request-{row.name}"),
        axis=1,
    )

    model_pathsafe_name = model.replace("/", "_")
    batch_jsonl_filename = (
        f"{grammar_name}_{model_pathsafe_name}_batched_{2*n_shots}-shot.jsonl"
    )
    batch_jsonl_path = grammar_path / batch_jsonl_filename
    log.info(f"Writing batch job to {batch_jsonl_path}")
    with open(batch_jsonl_path, "w") as f:
        for j in samples_df[f"{model}_batched_json"]:
            f.write(f"{j}\n")


def all_rand(
    # Grammar params
    h_low: int = 5,
    h_high: int = 100,
    lambda_: float = 0.01,
    # Sample params
    max_length: int = 50,
    samples_per_length: int = 10,
    gen_positive: bool = True,
    gen_negative: bool = True,
    max_tries_per_sample: int = 10,
    max_recursion_depth: int = 10000,
    pos_multiplier: int = 100,
    # Batch params
    models: list[str] = ["gpt-4o-mini", "gpt-4o", "o3-mini"],
    n_shots: list[int] = [0],
):
    n_terminals = int(min(max(random.expovariate(lambda_), h_low), h_high))
    n_nonterminals = int(min(max(random.expovariate(lambda_), h_low), h_high))
    n_lexical_rules = int(min(max(random.expovariate(lambda_), h_low), h_high))
    n_nonlexical_rules = int(min(max(random.expovariate(lambda_), h_low), h_high))

    log.info(
        f"Generating grammar with params: {n_terminals=}, {n_nonterminals=}, {n_lexical_rules=}, {n_nonlexical_rules=}"
    )

    grammar_dict = grammar(
        n_terminals=n_terminals,
        n_nonterminals=n_nonterminals,
        n_lexical_rules=n_lexical_rules,
        n_nonlexical_rules=n_nonlexical_rules,
    )

    if grammar_dict is None:
        log.warning("Failed to generate grammar")
        return None

    samples(
        grammar_name=grammar_dict["grammar_name"],
        max_length=max_length,
        samples_per_length=samples_per_length,
        gen_positive=gen_positive,
        gen_negative=gen_negative,
        max_tries_per_sample=max_tries_per_sample,
        max_recursion_depth=max_recursion_depth,
        pos_multiplier=pos_multiplier,
    )

    filtered_samples(
        grammar_name=grammar_dict["grammar_name"],
        max_length=max_length,
        samples_per_length=samples_per_length,
    )

    for model in models:
        for n_shot in n_shots:
            openai_batch(
                grammar_name=grammar_dict["grammar_name"],
                model=model,
                n_shots=n_shot,
            )


def all_grid(
    # Grammar params
    h_low: int = 10,
    h_high: int = 100,
    sweep_id: int = 0,
    # Sample params
    max_length: int = 50,
    samples_per_length: int = 10,
    gen_positive: bool = True,
    gen_negative: bool = True,
    max_tries_per_sample: int = 10,
    max_recursion_depth: int = 10000,
    pos_multiplier: int = 100,
    # Batch params
    models: list[str] = ["gpt-4o-mini", "gpt-4o", "o3-mini"],
    n_shots: list[int] = [0],
):
    # This will construct an HPARAM space of 256 different HPARAM combinations.

    hp_space = [range(h_low, h_high, 25)] * 4
    hp_space = list(itertools.product(*hp_space))
    HPSPACE_LEN = len(hp_space)

    hp_space = hp_space[sweep_id :: len(hp_space)]
    n_terminals, n_nonterminals, n_lexical_rules, n_nonlexical_rules = hp_space[0]

    log.info(f"Hyperparameter space: {hp_space}")
    log.info(f"Running sweep {sweep_id+1} of {HPSPACE_LEN}")

    log.info(
        f"Generating grammar with params: {n_terminals=}, {n_nonterminals=}, {n_lexical_rules=}, {n_nonlexical_rules=}"
    )

    grammar_dict = grammar(
        n_terminals=n_terminals,
        n_nonterminals=n_nonterminals,
        n_lexical_rules=n_lexical_rules,
        n_nonlexical_rules=n_nonlexical_rules,
    )

    if grammar_dict is None:
        log.warning("Failed to generate grammar")
        return None

    samples(
        grammar_name=grammar_dict["grammar_name"],
        max_length=max_length,
        samples_per_length=samples_per_length,
        gen_positive=gen_positive,
        gen_negative=gen_negative,
        max_tries_per_sample=max_tries_per_sample,
        max_recursion_depth=max_recursion_depth,
        pos_multiplier=pos_multiplier,
    )

    filtered_samples(
        grammar_name=grammar_dict["grammar_name"],
        max_length=max_length,
        samples_per_length=samples_per_length,
    )

    for model in models:
        for n_shot in n_shots:
            openai_batch(
                grammar_name=grammar_dict["grammar_name"],
                model=model,
                n_shots=n_shot,
            )


def xbar(
    lang: str = "english",
    n_samples: int = 2,
    verbose: bool = False,
):
    g_params: GrammarParams
    if lang == "english":
        g_params = fg_mxg.GrammarParams.english()
    elif lang == "german":
        g_params = fg_mxg.GrammarParams.german()
    else:
        raise ValueError(f"Unknown language: {lang}")

    if verbose:
        print("Running with params:")
        pprint.pprint(asdict(g_params))
    grammar_str: str = g_params.to_cfg_str()
    grammar: fg_grammar.Grammar = fg_grammar.Grammar.from_string(
        grammar_str, grammar_type=GType.CFG
    )
    if verbose:
        print(grammar.as_cfg)

    for _ in range(n_samples):
        s: dict[str, Any] = grammar.generate_tree()
        print(s["string"])


def scfg():
    en_de_params = fg_mxg.SyncGrammarParams.english_german()
    scfg_str = fg_mxg.generate_scfg(en_de_params)
    print(scfg_str)


if __name__ == "__main__":
    fire.Fire()
