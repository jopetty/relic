# formal-gym

Generating formal languages.

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

### Generating Grammars

```bash
uv run scripts/gen_grammar.py \
    --n_terminals=5 \
    --n_nonterminals=6 \
    --n_binary_rules=6 \
    --n_lexical_rules=5
```

This process may result in an empty final grammar, in which case you'll need to re-run this until a non-empty one is generated. Once a non-empty grammar is found, it will be saved to `data/grammars/` with as `sample_trim_DATETIME.cfg`.

### Generating Positive Samples

```bash
uv run scripts/gen_samples.py --grammar_file sample_trim_DATETIME.cfg
```
