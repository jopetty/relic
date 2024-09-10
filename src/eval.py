"""Evaluate models."""

import logging

import fire
import pyrootutils
from dotenv import load_dotenv
from utils.utils import get_logger

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%Y-%d-%m %H:%M:%S",
    level=logging.INFO,
)

log = get_logger(__name__)

PROJECT_ROOT = path = pyrootutils.find_root(
    search_from=__file__, indicator=".project-root"
)

load_dotenv()


def main():
    pass


if __name__ == "__main__":
    fire.Fire(main)
