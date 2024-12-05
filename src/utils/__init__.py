import argparse
import logging
import datetime
import os
from torch.optim import AdamW

from transformers import get_scheduler
from transformers.trainer_pt_utils import get_parameter_names
from transformers.pytorch_utils import ALL_LAYERNORM_LAYERS

from transformers import DataCollatorForLanguageModeling


def create_optimizer_scheduler(
    model, lr, max_steps, lr_scheduler_type, warmup_steps=50
):
    """
    Create AdamW optimizer and learning rate scheduler.
    """
    decay_parameters = get_parameter_names(model, ALL_LAYERNORM_LAYERS)
    decay_parameters = [name for name in decay_parameters if "bias" not in name]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if n in decay_parameters],
            "weight_decay": 0.01,
        },
        {
            "params": [
                p for n, p in model.named_parameters() if n not in decay_parameters
            ],
            "weight_decay": 0.00,
        },
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=lr, eps=1e-8)

    scheduler = get_scheduler(
        name=lr_scheduler_type,
        optimizer=optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=max_steps,
    )

    return optimizer, scheduler


def get_argparser():
    parser = argparse.ArgumentParser(description="Skill-It data selection")
    # data loading arguments
    parser.add_argument(
        "--task_name",
        type=str,
    )
    parser.add_argument(
        "--val_task_name",
        type=str,
    )
    parser.add_argument(
        "--get_grads",
        action="store_true",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        help="Dataset to use. Options: depth6, depth9, slimpj",
    )
    parser.add_argument(
        "--warm_start_path",
        type=str,
        help="Directory from which to load pre-trained outer model",
        default=None,
    )
    parser.add_argument(
        "--num_nodes",
        type=int,
        default=1,
    )
    parser.add_argument("--devices", type=int, default="-1")

    parser.add_argument(
        "--output_dir",
        type=str,
        default="./output/",
        help="Directory where all results and logs are stored.",
    )
    # general data sampling arguments
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
    )
    parser.add_argument("--lr", type=float, default=3e-4)
    # training arguments
    parser.add_argument(
        "--seq_length",
        type=int,
        default=2048,
    )
    parser.add_argument(
        "--n_epochs",
        default=1,
        type=int,
    )
    parser.add_argument(
        "--total_steps",
        type=int,
        default=-1,
        help="Maximum number of outer steps to train for",
    )
    parser.add_argument(
        "--model_name",
        default="EleutherAI/pythia-160M",
        type=str,
        help="Model to continually pre-train/fine-tune",
    )
    parser.add_argument(
        "--bsz",
        default=32,
        type=int,
    )
    parser.add_argument(
        "--eval_bsz",
        default=None,
        type=int,
    )
    # evaluation args
    parser.add_argument(
        "--num_ckpts",
        help="Number of checkpoints to evaluate the model at.",
        type=int,
        default=1,
    )
    parser.add_argument(
        "--eval_every",
        type=int,
        default=100,
    )
    parser.add_argument(
        "--lr_scheduler",
        type=str,
        default="cosine",
    )
    parser.add_argument(
        "--warmup_steps",
        type=int,
        default=50,
    )
    parser.add_argument(
        "--kl_loss",
        action="store_true",
    )
    parser.add_argument(
        "--kl_weight",
        type=float,
        default=0.1,
    )

    return parser


def get_logger(dir_path):
    # Create a logger
    logger = logging.getLogger("LLM-based evaluation")
    logger.setLevel(logging.INFO)

    # Create a file handler that writes to output.log
    file_handler = logging.FileHandler(os.path.join(dir_path, "output.log"))
    file_handler.setLevel(logging.INFO)

    # Create a stream handler that prints to the screen
    stream_handler = logging.StreamHandler()
    stream_handler.setLevel(logging.INFO)

    # Create a formatter for the log messages
    formatter = logging.Formatter(
        "%(asctime)s - %(levelname)s - %(filename)s - %(funcName)s - %(message)s"
    )
    file_handler.setFormatter(formatter)
    stream_handler.setFormatter(formatter)

    # Add the handlers to the logger
    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)

    logger.propagate = False

    return logger


def make_output_dir(output_dir):
    # run_id is MMDDYY
    run_id = datetime.datetime.now().strftime("%m%d%y")
    dir_path = os.path.join(output_dir, run_id)
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
    return dir_path
