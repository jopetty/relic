# xbar_schemas.py

from typing import Literal

from pydantic import BaseModel, model_validator


class RNGInfo(BaseModel):
    seed: int
    algorithm: Literal["python-random"]
    version: str


class GrammarParams(BaseModel):
    head_initial: bool
    spec_initial: bool
    pro_drop: bool
    proper_with_det: bool
    syllable_struct: str
    avg_syllables: int
    max_consonants: int


class Lexicon(BaseModel):
    verbs: list[str]
    nouns: list[str]
    propns: list[str]
    prons: list[str]
    adjs: list[str]
    det_def: list[str]
    det_indef: list[str]
    comps: list[str]
    tenses: list[str]
    asps: list[str]


class MonoGrammar(BaseModel):
    grammar_id: str
    format_version: str
    type: Literal["mono"]
    rng: RNGInfo
    parameters: GrammarParams
    lexicon: Lexicon


class SyncSide(BaseModel):
    parameters: GrammarParams
    lexicon: Lexicon


class SyncGrammar(BaseModel):
    grammar_id: str
    format_version: str
    type: Literal["sync"]
    rng: RNGInfo
    left_side: SyncSide
    right_side: SyncSide


Grammar = MonoGrammar | SyncGrammar


class SampleMonoOutput(BaseModel):
    output: str


class SampleSyncOutput(BaseModel):
    output: dict[Literal["left", "right"], str]


class Sample(BaseModel):
    sample_id: str
    grammar_id: str
    seed: int
    depth: int
    output: str | dict[Literal["left", "right"], str]

    @model_validator(mode="after")
    @classmethod
    def check_output_shape(cls, values):
        out = values.get("output")
        # allow either a simple string or a dict with both sides
        if not isinstance(out, str) and set(out.keys()) != {"left", "right"}:
            raise ValueError("sync samples must have both 'left' and 'right' outputs")
        return values
