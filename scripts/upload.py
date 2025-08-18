# upload.py
#
# Uploads a batch job to an LLM API for evaluation.

import datetime as dt
import logging
from pathlib import Path

import dotenv
import fire
from google import genai
from google.genai import types
import openai
import pyrootutils

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
grammars_dir = PROJECT_ROOT / "data" / "grammars"


def extract_batch_jsonl_path(
    grammar_name: str,
    grammar_path: Path,
    model: str = "gpt-4o-mini",
    n_shots: int = 0,
    eval_task: str = "accept",
):

    if eval_task == "accept":
        batch_jsonl_filename = f"{grammar_name}_{model}_batched_{2*n_shots}-shot.jsonl"
    elif eval_task == "generate":
        batch_jsonl_filename = f"{grammar_name}_{model}_batched_{2*n_shots}-shot_generate.jsonl"
    else:
        raise ValueError(f"Invalid evaluation task: {eval_task}. Must be one of ['accept', 'generate'].")

    batch_jsonl_path = grammar_path / batch_jsonl_filename

    # check that batch_jsonl_path exists
    if not batch_jsonl_path.exists():
        raise ValueError(f"Batch file {batch_jsonl_path} does not exist.")

    return batch_jsonl_path

def openai_batch(
    grammar_name: str,
    model: str = "gpt-4o-mini",
    n_shots: int = 0,
    eval_task: str = "accept",
):
    grammar_path = grammars_dir / f"{grammar_name}"

    batch_jsonl_path = extract_batch_jsonl_path(
        grammar_name,
        grammar_path,
        model,
        n_shots,
        eval_task,
    )

    log.info(f"Uploading OpenAI batch job from {batch_jsonl_path}")

    client = openai.OpenAI()
    batch_input_file = client.files.create(
        file=open(batch_jsonl_path, "rb"),
        purpose="batch",
    )

    log.info(f"Batch input file created: {batch_input_file}")

    batch_obj = client.batches.create(
        input_file_id=batch_input_file.id,
        endpoint="/v1/chat/completions",
        completion_window="24h",
        metadata={"description": "Batch job for grammar evaluation."},
    )

    log.info(f"Batch job created: {batch_obj}")

    timestamp = dt.datetime.now().strftime("%Y%m%d%H%M%S")
    log_file_path = grammar_path / f"{batch_obj.id}-{timestamp}.log"
    with open(log_file_path, "w") as f:
        f.write(f"{batch_input_file}\n\n")
        f.write(f"{batch_obj}")


def google_batch(
    grammar_name: str,
    model: str = "gemini-2.5-flash",
    n_shots: int = 0,
    eval_task: str = "accept",
):
    grammar_path = grammars_dir / f"{grammar_name}"

    batch_jsonl_path = extract_batch_jsonl_path(
        grammar_name,
        grammar_path,
        model,
        n_shots,
        eval_task,
    )

    log.info(f"Uploading Google batch job from {batch_jsonl_path}")

    client = genai.Client()
    batch_input_file = client.files.upload(
        file=batch_jsonl_path,
        config=types.UploadFileConfig(
            display_name=f"{grammar_name}_{model}_batched_{2*n_shots}-shot_{eval_task}.jsonl",
            mime_type="application/jsonl",
        ),
    )

    log.info(f"Batch input file created: {batch_input_file}")

    batch_obj = client.batches.create(
        model="gemini-2.5-flash",
        src=batch_input_file.name,
        config={
            'display_name': "Batch job for grammar evaluation.",
        },
    )

    timestamp = dt.datetime.now().strftime("%Y%m%d%H%M%S")
    batch_obj_pathsafe_name = batch_obj.name.replace("batches/", "batch_")

    batch_input_file_path = grammar_path / f"{batch_obj_pathsafe_name}_inputs.jsonl"
    with open(batch_jsonl_path, "rb") as src, open(batch_input_file_path, "wb") as dst:
        dst.write(src.read())

    log_file_path = grammar_path / f"{batch_obj_pathsafe_name}-{timestamp}.log"
    with open(log_file_path, "w") as f:
        f.write(f"{batch_input_file}\n\n")
        f.write(f"{batch_obj}")



if __name__ == "__main__":
    fire.Fire()
