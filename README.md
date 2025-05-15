# RELIC: Recognizing Languages In-Context

RELIC is an evaluation framework to measure how well large language models (LLMs) can
perform compositional instruction following by prompting models with a context-free
formal grammar (CFG) and a string generated from the grammar's terminal symbols and
asking the model to determine if the string is in the language defined by the CFG.

RELIC stochastically generates context-free grammars of a specified level of complexity,
and then generates positive and negative examples for each grammar. The synthetic
nature of the framework means that RELIC mitigates issues of dataset contamination
(since new datasets can be generated on the fly) and saturation (since the grammars
and examples can be generated with increasing complexity as models improve).

## Installation

1. Install `uv`.

```shell
curl -LsSf https://astral.sh/uv/install.sh | sh
```

2. Clone the repo.
3. Run with `uv`.

```shell
uv run src/sample.py  # Runs sample.py inside a virtual environment managed by uv.
```

## Usage

### Generating new datasets

To generate a new grammar, run

```shell
uv run scripts/generate.py grammar --n_terminals <int> --n_nonterminals <int> --n_lexical_rules <int> --n_nonlexical_rules <int>
```

where `n_terminals`, `n_nonterminals`, `n_lexical_rules`, and `n_nonlexical_rules` are
the generating hyperparameters for the grammar. This will create a new random grammar
with hyperparameters less than or equal to the specified values. The grammar will be
saved in `data/grammars/<grammar_name>/`.

### Generating new examples

To generate new examples from a grammar, run

```shell
uv run scripts/generate.py samples --grammar_name <str> --max_length <int> --samples_per_length <int>
```

See the method definition for additional arguments.

To filter existing examples to include on a fixed number of examples per length and type,
run

```shell
uv run scripts/generate.py filter_samples --grammar_name <str> --max_length <int> --samples_per_length <int>
```

### Generating batch files for evaluation

To generate a batch file compatible with OpenAI's batch API, run

```shell
uv run scripts/generate.py openai_batch --grammar_name <str> --model <str> (--subsample_n <int>)
```

where `model` is the name of the model to use (e.g., `o4-mini`) and `subsample_n` is the
number of examples per type per length to include in the batch file. If this argument
is not passed, all examples from the filtered set will be included.

## RELIC-400

As a demonstration of the framework, we include a set of 200 grammars called RELIC-400,
whose hyperparameters are all bounded above by 400. These grammars are already present,
along with filtered examples, in the `data/grammars/` directory.
