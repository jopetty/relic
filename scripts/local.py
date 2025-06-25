# local.py
#
# Runs a local LLM evaluation.

import hashlib
import logging
from pathlib import Path
from typing import Any

import datasets
import dotenv
import fire
import pyrootutils
import torch
import transformers
from tqdm import tqdm

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
    evaluation: str = "",
    # Model parameters
    model: str = "google/gemma-3-1b-it",
    attn_implementation: str = "sdpa",
    torch_dtype: str = "torch.bfloat16",
    device_map: str = "auto",
    do_compile: bool = True,
    compile_mode: str = "default",
    # Generation parameters
    max_new_tokens: int = 2_000,
    do_sample: bool = True,
    batch_size: int = 16,
):
    # Build dict of all parameters
    params: dict[str, Any] = {
        "grammar_name": grammar_name,
        "n_shots": n_shots,
        "model": model,
        "attn_implementation": attn_implementation,
        "torch_dtype": torch_dtype,
        "device_map": device_map,
        "max_new_tokens": max_new_tokens,
        "do_sample": do_sample,
        "batch_size": batch_size,
        "compile": do_compile,
        "compile_mode": compile_mode,
    }
    log.info(f"Running local inference with {params=}")

    grammars_dir: Path = PROJECT_ROOT / "data" / "grammars"
    grammar_path: Path = grammars_dir / f"{grammar_name}"

    model_pathsafe_name: str = model.replace("/", "_")
    if evaluation:
        evaluation = f"_{evaluation}"
    batch_jsonl_filename: str = f"{grammar_name}_{model_pathsafe_name}_batched_{2 * n_shots}-shot{evaluation}.jsonl"
    batch_jsonl_path: Path = grammar_path / batch_jsonl_filename

    batch_id_hash: str = hashlib.md5(str(batch_jsonl_filename).encode()).hexdigest()
    batch_id: str = f"batch_{batch_id_hash}"

    results_filename: str = f"{batch_id}_results.jsonl"
    results_path: Path = grammar_path / results_filename
    inputs_filename: str = f"{batch_id}_inputs.jsonl"
    inputs_path: Path = grammar_path / inputs_filename

    if not batch_jsonl_path.exists():
        raise ValueError(f"Batch file {batch_jsonl_path} does not exist.")

    log.info(f"Running local evaluation from {batch_jsonl_path}")

    # Load the dataset
    dataset: datasets.DatasetDict = datasets.load_dataset(
        "json", data_files=str(batch_jsonl_path)
    )
    dataset: datasets.Dataset = dataset["train"]

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

    dataset: datasets.Dataset = (
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
    dataset: datasets.Dataset = dataset.map(create_response).remove_columns(["body"])

    log.info(f"Loading model {model=}")

    # Determine device
    device: str = (
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu"
    )

    log.info(f"Using device: {device}")

    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model,
        use_fast=True,
        padding_side="left",
    )

    torch_dtype_val = (
        torch.bfloat16
        if torch_dtype == "torch.bfloat16"
        else torch.float16
        if torch_dtype == "torch.float16"
        else "auto"
    )

    model = transformers.AutoModelForCausalLM.from_pretrained(
        model,
        torch_dtype=torch_dtype_val,
        device_map=device_map,
        attn_implementation=attn_implementation,
    )

    if do_compile:
        model = torch.compile(model, mode=compile_mode)

    generation_config = transformers.GenerationConfig(
        max_new_tokens=max_new_tokens,
        do_sample=True,
    )

    log.info(f"Loaded model {model=}")
    log.info(f"Generating with {generation_config=}")

    log.info("Tokenzing inputs")
    tokenized_inputs = tokenizer(
        dataset["prompt"],
        return_tensors="pt",
        padding=True,
        pad_to_multiple_of=8,
    )
    tokenized_inputs = {k: v.to(model.device) for k, v in tokenized_inputs.items()}

    model.eval()

    log.info("Starting generation...")

    results = []
    for i in tqdm(range(0, len(dataset), batch_size), desc="Generating responses"):
        inputs = {
            "input_ids": tokenized_inputs["input_ids"][i : i + batch_size],
            "attention_mask": tokenized_inputs["attention_mask"][i : i + batch_size],
        }

        # Generate responses
        with torch.inference_mode():
            outputs = model.generate(
                **inputs,
                generation_config=generation_config,
            )

        # Decode generated tokens, skipping special tokens and prompt
        # We need to slice the output tensors to only get the generated part
        generated_ids = outputs[:, inputs["input_ids"].shape[1] :]
        batch_responses = tokenizer.batch_decode(
            generated_ids,
            skip_special_tokens=True,
        )

        # Store results for this batch
        for j, response_text in enumerate(batch_responses):
            original_example = dataset[i + j]
            old_response = original_example["response"]
            old_response["body"]["choices"] = [
                {"message": {"content": response_text.strip()}}
            ]
            results.append(
                {
                    **original_example,  # Keep original fields
                    "response": old_response,
                }
            )

    del model
    del tokenizer

    # Create a new dataset from the results
    if results:
        final_columns = list(results[0].keys())
        if "prompt" in final_columns:
            final_columns.remove("prompt")
        # Convert list of dicts to dict of lists for datasets.Dataset.from_dict
        results_dict = {col: [item[col] for item in results] for col in final_columns}
        processed_dataset = datasets.Dataset.from_dict(results_dict)
    else:
        raise ValueError("No results generated. Check the model and dataset.")

    log.info(f"Writing responses to {results_path}")
    processed_dataset.to_json(str(results_path), lines=True)

    del processed_dataset


if __name__ == "__main__":
    fire.Fire(run)
