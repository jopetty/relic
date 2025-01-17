"""
Save average activations of the model on a dataset
"""

import fire
import json
import os

from tqdm import tqdm
import pickle
import glob

import torch
from datasets import load_from_disk
from torch.utils.data import DataLoader
from transformers import DataCollatorForLanguageModeling, AutoTokenizer

from modeling_ppt_neox import PPTNeoXModel


def save_avg_blimp_activation(
    dataset, initialize_from, max_num_examples=4096, blimp=False
):
    """
    If items are of different lengths, batch size should be 1
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = (PPTNeoXModel.from_pretrained(initialize_from)).to(device)
    model.reset_read_avg_activation()

    dataset = load_from_disk(dataset)
    dataset = dataset.shuffle(seed=0).select(range(min(max_num_examples, len(dataset))))

    @torch.inference_mode()
    def do_inferences(
        model, good_input_ids, good_attention_mask, bad_input_ids, bad_attention_mask
    ):
        model(
            input_ids=torch.tensor(good_input_ids).to(device),
            attention_mask=torch.tensor(good_attention_mask).to(device),
        )
        model(
            input_ids=torch.tensor(bad_input_ids).to(device),
            attention_mask=torch.tensor(bad_attention_mask).to(device),
        )

    dataset = dataset.map(
        lambda ex: do_inferences(
            model,
            ex["good_input_ids"],
            ex["good_attention_mask"],
            ex["bad_input_ids"],
            ex["bad_attention_mask"],
        ),
        batched=False,
    )

    avg_activations = {}
    for n, m in model.named_modules():
        if hasattr(m, "get_avg_activation"):
            avg_activations[n] = m.get_avg_activation().cpu().numpy()
            print(n)
            print(avg_activations[n])
            print()

    output_path = os.path.join(initialize_from, "avg_activations.pkl")
    pickle.dump(avg_activations, open(output_path, "wb+"))


def save_avg_activation(
    dataset, initialize_from, bsz=16, max_num_examples=4096, blimp=False
):
    """
    If items are of different lengths, batch size should be 1
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = (PPTNeoXModel.from_pretrained(initialize_from)).to(device)
    model.reset_read_avg_activation()

    dataset = load_from_disk(dataset)
    dataset = dataset.shuffle(seed=0).select(range(min(max_num_examples, len(dataset))))

    @torch.inference_mode()
    def do_inferences(model, input_ids, attention_mask):
        model(
            input_ids=torch.tensor(input_ids).to(device),
            attention_mask=torch.tensor(attention_mask).to(device),
        )

    avg_activations = {}
    for n, m in model.named_modules():
        if hasattr(m, "get_avg_activation"):
            avg_activations[n] = m.get_avg_activation().cpu().numpy()
            print(n)
            print(avg_activations[n])
            print()

    output_path = os.path.join(initialize_from, "avg_activations.pkl")
    pickle.dump(avg_activations, open(output_path, "wb+"))


def main(super_dir, bsz, blimp=True):
    for model_dir in glob.glob(super_dir + "/*"):
        print(model_dir)
        if blimp:
            save_avg_blimp_activation(
                dataset="./data/tokenized/blimp",
                initialize_from=model_dir,
            )
        else:
            save_avg_activation(
                dataset="./data/tokenized/blimp",
                initialize_from=model_dir,
                bsz=bsz,
            )


if __name__ == "__main__":
    fire.Fire(main)
