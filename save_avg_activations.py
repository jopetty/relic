"""
Save average activations of the model on a dataset
"""

import fire
import json
import os

from tqdm import tqdm
import pickle

import torch
from datasets import load_from_disk
from torch.utils.data import DataLoader
from transformers import DataCollatorForLanguageModeling, AutoTokenizer

from modeling_ppt_neox import PPTNeoXModel


def main(dataset, initialize_from, output_path, max_num_examples=4096):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = (PPTNeoXModel.from_pretrained(initialize_from)).to(device)
    tokenizer = AutoTokenizer.from_pretrained(initialize_from)
    model.reset_read_avg_activation()

    dataset = load_from_disk(dataset)
    dataset = dataset.select(range(min(max_num_examples, len(dataset)))).remove_columns(
        ["text"]
    )

    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer, mlm=False, pad_to_multiple_of=2048
    )
    dataloader = DataLoader(dataset, batch_size=16, collate_fn=data_collator)

    for batch in tqdm(dataloader):
        model(
            input_ids=batch["input_ids"].to(device),
            attention_mask=batch["attention_mask"].to(device),
        )

    avg_activations = {}
    for n, m in model.named_modules():
        if hasattr(m, "get_avg_activation"):
            avg_activations[n] = m.get_avg_activation().cpu().numpy()
            print(n)
            print(avg_activations[n])
            print()

    pickle.dump(avg_activations, open(output_path, "wb+"))


if __name__ == "__main__":
    fire.Fire(main)
