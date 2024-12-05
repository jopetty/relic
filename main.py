from transformers import AutoTokenizer, set_seed
import numpy as np
import wandb

import torch
import warnings


from src.utils import (
    get_logger,
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
        "--synth_dir_train",
        type=str,
        help="Directory from which to load synthetic training data",
        default="./data/tokenized/depth6_train",
    )
    parser.add_argument(
        "--synth_dir_val",
        type=str,
        help="Directory from which to load synthetic validation data",
        default="./data/tokenized/depth6_val",
    )
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

    # load datasets or dataset generators, depending on the task
    if args.dataset == "depth6" or args.dataset == "depth9":
        train_data = HybridDataset(
            args.seed,
            is_eval=False,
            sampling_prob=args.sampling_p,
            seq_len=2048,
            synthetic_dir=args.synth_dir_train,
        )
        validation_data = HybridDataset(
            args.seed,
            is_eval=True,
            sampling_prob=args.sampling_p,
            seq_len=2048,
            synthetic_dir=args.synth_dir_val,
        )
    else:
        train_data = get_slimpj_dataset(args.seed, is_eval=False, seq_len=2048)
        validation_data = get_slimpj_dataset(args.seed, is_eval=True, seq_len=2048)

    trainer = StaticTrainer(
        args,
        tokenizer,
    )

    trainer.train(
        train_data,
        validation_data,
    )

    # save
    trainer.model.save_pretrained(output_dir_path)


if __name__ == "__main__":
    main()
