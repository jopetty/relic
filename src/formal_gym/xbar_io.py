# xbar_io.py

from pathlib import Path
from typing import Iterator

import pyrootutils
from pydantic import TypeAdapter

from formal_gym.metaxbargrammar import GrammarParams, SyncGrammarParams
from formal_gym.metaxbargrammar import XBarGrammar as XBarCFG
from formal_gym.scfg import SCFG as XBarSCFG
from formal_gym.xbar_schemas import Grammar as GrammarSchema

PROJECT_ROOT: Path = pyrootutils.find_root(
    search_from=__file__, indicator=".project-root"
)


def _write_grammar(grammar: GrammarSchema, path: Path):
    """Append a single grammar to a file."""
    with path.open("a", encoding="utf-8") as f:
        f.write(grammar.model_dump_json(exclude_none=True) + "\n")


def _read_grammars(path: Path) -> Iterator[GrammarSchema]:
    """Read grammars from a file."""
    adapter: TypeAdapter[GrammarSchema] = TypeAdapter(GrammarSchema)
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            yield adapter.validate_json(line)


def load_grammars() -> Iterator[GrammarSchema]:
    """Load grammars from the xbar directory."""
    grammar_dir: Path = PROJECT_ROOT / "xbar"
    grammars_file: Path = grammar_dir / "grammars.jsonl"
    return _read_grammars(grammars_file)


def save_grammar(
    grammar: GrammarSchema | GrammarParams | SyncGrammarParams | XBarCFG | XBarSCFG,
):
    """Save a grammar to the xbar directory."""
    grammar_dir: Path = PROJECT_ROOT / "xbar"
    grammars_file: Path = grammar_dir / "grammars.jsonl"

    if isinstance(grammar, GrammarSchema):
        _write_grammar(grammar, grammars_file)
    elif isinstance(grammar, GrammarParams):
        # TODO:
        # To make this work we need to refactor XBarCFG & XBarSCFG & GrammarParams &c
        # to include grammar_id, format_version, rng, and lexicon
        mono_grammar: GrammarSchema = GrammarSchema(
            grammar_id=grammar.grammar_id,
            format_version=grammar.format_version,
            type="mono",
            rng=grammar.rng,
            parameters=grammar,
            lexicon=grammar.lexicon,
        )
        _write_grammar(mono_grammar, grammars_file)
