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
from tqdm import tqdm  # Add tqdm for progress bar

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
    max_new_tokens: int = 200,
    batch_size: int = 2,
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

    log.info(f"Loading model {model}")

    # Determine device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    log.info(f"Using device: {device}")

    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model,
        use_fast=True,
        padding_side="left",
    )
    # Set pad token if it doesn't exist
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        log.info("Setting pad_token to eos_token")

    model = transformers.AutoModelForCausalLM.from_pretrained(
        model,
        torch_dtype="auto",
        device_map="auto",
    )

    model.eval()  # Set model to evaluation mode

    log.info("Starting generation...")

    results = []
    # Process dataset in batches
    for i in tqdm(range(0, len(dataset), batch_size), desc="Generating responses"):
        batch_prompts = dataset[i : i + batch_size]["prompt"]

        # Tokenize the batch
        inputs = tokenizer(
            batch_prompts,
            return_tensors="pt",
            padding=True,
        ).to(model.device)  # Move inputs to the same device as the model

        # Generate responses
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )

        # Decode generated tokens, skipping special tokens and prompt
        # We need to slice the output tensors to only get the generated part
        generated_ids = outputs[:, inputs.input_ids.shape[1] :]
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

    # Create a new dataset from the results
    # Ensure all columns from the original dataset (except 'prompt') are included
    # Get columns from the first result item, excluding 'prompt'
    if results:
        final_columns = list(results[0].keys())
        if "prompt" in final_columns:
            final_columns.remove("prompt")
        # Convert list of dicts to dict of lists for datasets.Dataset.from_dict
        results_dict = {col: [item[col] for item in results] for col in final_columns}
        processed_dataset = datasets.Dataset.from_dict(results_dict)
    else:
        # Handle empty results case
        processed_dataset = dataset.remove_columns(
            ["prompt"]
        )  # Or create an empty dataset with correct schema

    log.info(f"Writing responses to {results_path}")
    processed_dataset.to_json(str(results_path), lines=True)

    del model
    del tokenizer
    del processed_dataset


if __name__ == "__main__":
    fire.Fire(run)
