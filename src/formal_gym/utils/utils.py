import logging
import os
import random

import numpy as np
import pkg_resources

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%Y-%d-%m %H:%M:%S",
    level=logging.INFO,
)


def get_logger(name=__name__) -> logging.Logger:
    logger = logging.getLogger(name)
    return logger


log = get_logger(__name__)


def set_all_seeds(seed: int):
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    random.seed(seed)

    installed = {pkg.key for pkg in pkg_resources.working_set}
    if "torch" in installed:
        import torch

        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    log.info(f"All seeds set to {seed}.")
