# local.py
#
# Runs a local LLM evaluation.

import hashlib
import logging

import datasets
import dotenv
import fire
import pyrootutils
import torch
import transformers

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

dotenv.load_dotenv(PROJECT_ROOT / ".env")


def run(
    # Grammar parameters
    grammar_name: str,
    n_shots: int = 0,
    # Model parameters
    model: str = "google/gemma-2-2b-it",
    # Pipeline parameters
    max_new_tokens: int = None,
    batch_size: int = 10,
):
    grammars_dir = PROJECT_ROOT / "data" / "grammars"
    grammar_path = grammars_dir / f"{grammar_name}"

    model_pathsafe_name = model.replace("/", "_")
    batch_jsonl_filename = (
        f"{grammar_name}_{model_pathsafe_name}_batched_{2*n_shots}-shot.jsonl"
    )
    batch_jsonl_path = grammar_path / batch_jsonl_filename

    batch_id_hash = hashlib.md5(str(batch_jsonl_filename).encode()).hexdigest()
    batch_id = f"batch_{batch_id_hash}"

    results_filename = f"{batch_id}_results.jsonl"
    results_path = grammar_path / results_filename
    inputs_filename = f"{batch_id}_inputs.jsonl"
    inputs_path = grammar_path / inputs_filename

    if not batch_jsonl_path.exists():
        raise ValueError(f"Batch file {batch_jsonl_path} does not exist.")

    log.info(f"Running local evaluation from {batch_jsonl_path}")

    # Load the dataset
    dataset = datasets.load_dataset("json", data_files=str(batch_jsonl_path))
    dataset = dataset["train"]

    def flatten_body(example):
        return example["body"]

    def flatten_messages(example):
        return example["messages"][0]

    def flatten_metadata(example):
        example["body"]["metadata"] = example["metadata"]
        return example

    def create_input(example):
        example["prompt"] = example["content"]
        return example

    def create_response(example):
        example["response"] = {"body": example["body"]}
        return example

    dataset = (
        dataset.map(flatten_body)
        .map(flatten_messages)
        .map(flatten_metadata)
        .map(create_input)
        .remove_columns(
            [
                "method",
                "url",
                "store",
                "model",
                "messages",
                "content",
                "role",
                "metadata",
            ]
        )
    )

    log.info(f"Dataset loaded: {dataset}")
    log.info(f"Writing inputs to {inputs_path}")
    dataset.to_json(str(inputs_path), lines=True)

    # To match the format of the OpenAI responses, we need to take all the `body`
    # fields and put them inside a `response` field.
    dataset = dataset.map(create_response).remove_columns(["body"])

    # subsample
    dataset = dataset.select(range(10))

    log.info(f"Loading model {model}")

    pipe = transformers.pipeline(
        "text-generation",
        model=model,
        model_kwargs={"torch_dtype": torch.bfloat16},
        device_map="auto",
        return_full_text=False,
        batch_size=batch_size,
        max_new_tokens=max_new_tokens,
    )

    outputs = iter(pipe((_ for _ in dataset["prompt"])))

    def get_response(example):
        output = next(outputs)[0]["generated_text"].strip()
        old_response = example["response"]
        old_response["body"]["choices"] = [{"message": {"content": output}}]
        example["response"] = old_response
        return example

    dataset = dataset.map(
        get_response,
        desc="get_response",
        batch_size=batch_size,
    ).remove_columns(["prompt"])
    log.info(f"Writing responses to {results_path}")
    dataset.to_json(str(results_path), lines=True)

    del pipe
    del dataset


if __name__ == "__main__":
    fire.Fire()
