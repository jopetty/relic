"""
Save average activations of the model on a dataset
"""

import fire
import os

from tqdm import tqdm
import pickle
import glob

import torch
from datasets import load_from_disk, load_dataset
from transformers import AutoTokenizer

from modeling_ppt_neox import PPTNeoXModel


def save_avg_blimp_activation(initialize_from, max_num_examples=2048):
    """
    If items are of different lengths, batch size should be 1
    """
    if os.path.exists(os.path.join(initialize_from, "avg_activations_blimp.pkl")):
        print("Already exists")
        return

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = (PPTNeoXModel.from_pretrained(initialize_from)).to(device)
    model.reset_read_avg_activation()

    dataset = load_from_disk("./data/tokenized/blimp")
    dataset = dataset.shuffle(seed=0).select(range(min(max_num_examples, len(dataset))))

    @torch.inference_mode()
    def do_inferences(
        model, good_input_ids, good_attention_mask, bad_input_ids, bad_attention_mask
    ):
        model(
            input_ids=torch.tensor(good_input_ids).unsqueeze(0).to(device),
            attention_mask=torch.tensor(good_attention_mask).unsqueeze(0).to(device),
        )
        model(
            input_ids=torch.tensor(bad_input_ids).unsqueeze(0).to(device),
            attention_mask=torch.tensor(bad_attention_mask).unsqueeze(0).to(device),
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

    output_path = os.path.join(initialize_from, "avg_activations_blimp.pkl")
    pickle.dump(avg_activations, open(output_path, "wb+"))


def save_avg_activation(
    initialize_from,
    data_dir,
    bsz=16,
    max_num_examples=2048,
):
    """
    If items are of different lengths, batch size should be 1
    """
    if os.path.exists(os.path.join(initialize_from, "avg_activations.pkl")):
        print("Already exists")
        return

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = (PPTNeoXModel.from_pretrained(initialize_from)).to(device)
    model.reset_read_avg_activation()

    if data_dir == "c4":
        dataset = load_dataset(
            "json",
            data_files=[
                f"/vast/work/public/ml-datasets/c4/en/c4-validation.0000{i}-of-00008.json"
                for i in range(8)
            ],
            split="train",
        )
        tokenizer = AutoTokenizer.from_pretrained(
            "EleutherAI/pythia-160m",
        )
        tokenizer.add_special_tokens({"pad_token": "<|padding|>"})
        dataset = (
            dataset.shuffle(seed=0)
            .select(range(min(max_num_examples, len(dataset))))
            .map(
                lambda ex: tokenizer(ex["text"], padding="max_length", truncation=True),
            )
        )
        bsz = 1  # no padding
    else:
        dataset = load_from_disk(data_dir)
        dataset = dataset.shuffle(seed=0).select(
            range(min(max_num_examples, len(dataset)))
        )

    @torch.inference_mode()
    def do_inferences(model, input_ids, attention_mask):
        model(
            input_ids=torch.tensor(input_ids).to(device),
            attention_mask=torch.tensor(attention_mask).to(device),
        )

    dataset = dataset.map(
        lambda ex: do_inferences(model, ex["input_ids"], ex["attention_mask"]),
        batched=True,
        batch_size=bsz,
    )

    avg_activations = {}
    for n, m in model.named_modules():
        if hasattr(m, "get_avg_activation"):
            avg_activations[n] = m.get_avg_activation().cpu().numpy()

    output_path = os.path.join(initialize_from, "avg_activations.pkl")
    pickle.dump(avg_activations, open(output_path, "wb+"))


def main(super_dir, data_dir=None, bsz=32, blimp=True):
    for model_dir in glob.glob(super_dir + "/*/"):
        print(model_dir)
        if blimp:
            save_avg_blimp_activation(
                initialize_from=model_dir,
            )
        else:
            save_avg_activation(
                data_dir=data_dir,
                initialize_from=model_dir,
                bsz=bsz,
            )


if __name__ == "__main__":
    fire.Fire(
        {
            "save_avg_activation": save_avg_activation,
            "save_avg_blimp_activation": save_avg_blimp_activation,
            "main": main,
        }
    )
