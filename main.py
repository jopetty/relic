from transformers import AutoTokenizer, set_seed
import numpy as np
import wandb
import os
import torch

from src.utils import (
    get_argparser,
    make_output_dir,
)
from src.data.data import HybridDataset
from src.data.utils import get_slimpj_dataset
from src.trainer import StaticTrainer


def main():
    np.set_printoptions(precision=3, suppress=True)
    torch.set_float32_matmul_precision("medium")

    parser = get_argparser()
    parser.add_argument(
        "--sampling_p",
        type=float,
        default=0.5,
        help="Probability of sampling from synthetic data",
    )

    args = parser.parse_args()
    wandb.init(project="synthetic-midtraining", config=args)
    set_seed(args.seed)

    output_dir_path = make_output_dir(args.output_dir)

    # logger = get_logger(output_dir_path)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    tokenizer.pad_token = tokenizer.eos_token

    dataset_to_path = {
        "depth6": "./data/tokenized/depth6",
        "depth9": "./data/tokenized/depth9",
    }

    # load datasets or dataset generators, depending on the task
    if args.dataset == "depth6" or args.dataset == "depth9":
        train_data = HybridDataset(
            args.seed,
            is_eval=False,
            sampling_prob=args.sampling_p,
            seq_len=2048,
            synthetic_dir=dataset_to_path[args.dataset],
        )
        validation_data = HybridDataset(
            args.seed,
            is_eval=True,
            sampling_prob=args.sampling_p,
            seq_len=2048,
            synthetic_dir=dataset_to_path[args.dataset],
        )
    else:
        train_data = get_slimpj_dataset(args.seed, is_eval=False, seq_len=2048)
        validation_data = get_slimpj_dataset(args.seed, is_eval=True, seq_len=2048)

    trainer = StaticTrainer(
        args,
        tokenizer,
        output_dir_path,
    )

    trainer.train(
        train_data,
        validation_data,
    )


if __name__ == "__main__":
    main()
