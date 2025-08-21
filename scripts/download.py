# download.py
#
# Downloads a batch job from an LLM API for analysis.

import logging
from pathlib import Path
import re
from typing import Literal

import dotenv
import fire
from google import genai
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


def download_batch(
    grammar_name: str,
    batch_format: Literal["openai", "google"] = "openai",
    batch_id: str | None = None,
):
    grammars_dir = PROJECT_ROOT / "data" / "grammars"
    grammar_path = grammars_dir / f"{grammar_name}"

    if batch_format == "openai":
        openai_batch(grammar_path, batch_id)
    elif batch_format == "google":
        google_batch(grammar_path, batch_id)
    else:
        raise ValueError(f"Invalid batch format: {batch_format}")

def openai_batch(
    grammar_path: Path,
    batch_id: str | None = None,
):
    
    if batch_id is None:
        batch_logs = list(grammar_path.glob("batch_*.log"))
        batch_id_re = re.compile(r"(batch_[a-zA-Z0-9]+)")
        batch_ids = [batch_id_re.search(f.name).group(1) for f in batch_logs]
    else:
        batch_ids = [batch_id]

    if not batch_ids:
        log.info("No undownloaded batch jobs found.")
    else:
        client = openai.OpenAI()
        for batch_id in batch_ids:
            input_path = grammar_path / f"{batch_id}_inputs.jsonl"
            output_path = grammar_path / f"{batch_id}_results.jsonl"

            if not (input_path.exists() and output_path.exists()):
                batch_results = client.batches.retrieve(batch_id)

                if batch_results.status == "completed":
                    log.info(batch_results)

                    input_file = client.files.content(batch_results.input_file_id)
                    output_file = client.files.content(batch_results.output_file_id)

                    log.info(f"Writing batch inputs to {input_path}")
                    with open(input_path, "w") as f:
                        f.write(input_file.text)
                    log.info(f"Writing batch results to {output_path}")
                    with open(output_path, "w") as f:
                        f.write(output_file.text)
                else:
                    log.warning("Batch job not completed yet.")

def google_batch(
    grammar_path: Path,
    batch_id: str | None = None,
):
    
    if batch_id is None:
        batch_logs = list(grammar_path.glob("batch_*.log"))
        batch_id_re = re.compile(r"(batch_[a-zA-Z0-9]+)")
        batch_ids = [batch_id_re.search(f.name).group(1) for f in batch_logs]
    else:
        batch_ids = [batch_id]

    if not batch_ids:
        log.info("No undownloaded batch jobs found.")
    else:
        client = genai.Client()
        for batch_id in batch_ids:
            output_path = grammar_path / f"{batch_id}_results.jsonl"

            if not output_path.exists():
                batch_results = client.batches.get(name=batch_id.replace("batch_", "batches/"))

                if batch_results.state.name == "JOB_STATE_SUCCEEDED":
                    log.info(batch_results)

                    output_file = client.files.download(file=batch_results.dest.file_name)
                    log.info(f"Writing batch results to {output_path}")
                    with open(output_path, "w") as f:
                        f.write(output_file.decode("utf-8"))
                else:
                    log.warning("Batch job not completed yet.")


if __name__ == "__main__":
    fire.Fire()
