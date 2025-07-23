# upload.py
#
# Uploads a batch job to an LLM API for evaluation.

import datetime as dt
import logging

import dotenv
import fire
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


def openai_batch(
    grammar_name: str,
    model: str = "gpt-4o-mini",
    n_shots: int = 0,
    eval_task: str = "accept",
):
    grammars_dir = PROJECT_ROOT / "data" / "grammars"
    grammar_path = grammars_dir / f"{grammar_name}"

    if eval_task == "accept":
        batch_jsonl_filename = f"{grammar_name}_{model}_batched_{2*n_shots}-shot_accept.jsonl"
    elif eval_task == "generate":
        batch_jsonl_filename = f"{grammar_name}_{model}_batched_{2*n_shots}-shot_generate.jsonl"
    else:
        raise ValueError(f"Invalid evaluation task: {eval_task}. Must be one of ['accept', 'generate'].")

    batch_jsonl_path = grammar_path / batch_jsonl_filename

    # check that batch_jsonl_path exists
    if not batch_jsonl_path.exists():
        raise ValueError(f"Batch file {batch_jsonl_path} does not exist.")

    log.info(f"Uploading batch job from {batch_jsonl_path}")

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


if __name__ == "__main__":
    fire.Fire()
