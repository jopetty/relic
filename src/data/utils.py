import datasets
from datasets import load_dataset, IterableDataset, load_from_disk, Features
import os
from transformers import AutoTokenizer

# from unsloth import FastLanguageModel
import numpy as np
from tqdm import tqdm
import torch


REDPAJAMA_DATA_PATH = os.path.join("/scratch/myh2014/data/", "datasets/slim_6b_pythia/")
DATA_PATHS = {
    "crasp_min_first": "/scratch/myh2014/formal-gym/data/tokenized/depth9_crasp_min_first",
    "crasp_min_second": "/scratch/myh2014/formal-gym/data/tokenized/depth9_crasp_min_second",
    "crasp_first": "/scratch/myh2014/formal-gym/data/tokenized/depth9_crasp_first",
    "crasp_second": "/scratch/myh2014/formal-gym/data/tokenized/depth9_crasp_second",
    "fom_first": "/scratch/myh2014/formal-gym/data/tokenized/depth9_fom_first",
    "fom_second": "/scratch/myh2014/formal-gym/data/tokenized/depth9_fom_second",
}


def load_text_files(file_dir):
    texts = []
    # Read each file in the directory
    for filename in tqdm(os.listdir(file_dir)):
        if filename.endswith(".txt"):  # Adjust file extension if needed
            file_path = os.path.join(file_dir, filename)
            with open(file_path, "r", encoding="utf-8") as f:
                texts.append(f.read())

    return datasets.Dataset.from_dict({"text": texts})


def cache_data(
    file_dir: str, out_dir: str, tokenizer_name: str = "EleutherAI/pythia-160m"
):
    # load text files into huggingface dataset
    dataset = load_text_files(file_dir)
    # initialize tokenizer
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    tokenizer.add_special_tokens({"pad_token": "<|padding|>"})

    # silly hack
    # _, tokenizer = FastLanguageModel.from_pretrained(
    #     model_name=tokenizer_name,
    #     max_seq_length=2048,
    #     dtype=None,
    # )
    dataset = dataset.map(
        lambda x: tokenizer(
            x["text"], padding="max_length", truncation=True, max_length=2048
        ),
        batched=True,
    )

    dataset.save_to_disk(out_dir)


def cache_data_from_hf(
    dataset_name: str, out_dir: str, tokenizer_name: str = "EleutherAI/pythia-160m"
):
    # no padding, allow DataCollatorWithFlatten to pad
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    tokenizer.add_special_tokens({"pad_token": "<|padding|>"})

    if "code" in dataset_name:
        dataset = load_dataset(dataset_name)
        dataset = dataset.map(
            lambda x: tokenizer(x["code"], truncation=True, max_length=2048),
            batched=True,
        ).remove_columns(
            [
                "repo",
                "path",
                "url",
                "code",
                "code_tokens",
                "docstring",
                "docstring_tokens",
                "language",
                "partition",
                "avg_line_len",
            ]
        )
    elif "goat" in dataset_name:
        # concatenate instruction and ouptut columns
        dataset = load_dataset(dataset_name)
        dataset = dataset.map(
            lambda x: tokenizer(
                x["instruction"] + " " + x["output"],
                truncation=True,
                max_length=2048,
            ),
        ).remove_columns(["instruction", "output", "input", "answer"])
    elif dataset_name == "cogs":
        dataset = load_dataset(
            "csv",
            data_files="./data/cogs_train.tsv",
            delimiter="\t",
            column_names=["input", "output", "generalization"],
            split="train",
        )
        dataset = dataset.map(
            lambda x: tokenizer(
                x["input"] + " " + x["output"],
                truncation=True,
                max_length=2048,
            ),
        ).remove_columns(["generalization"])
    elif dataset_name == "s5":
        dataset = load_dataset(
            "csv",
            data_files="./data/s5.csv",
            split="train",
            header=0,
        )
        dataset = dataset.map(
            lambda x: tokenizer(
                x["input"] + " " + x["target"],
                truncation=True,
                max_length=2048,
            ),
        ).remove_columns(["seed", "input", "target"])
    elif dataset_name == "dyck":
        dataset = load_dataset(
            "text", data_files="./data/dyck_sequences.txt", split="train"
        )
        dataset = dataset.map(
            lambda x: tokenizer(
                x["text"],
                truncation=True,
                max_length=2048,
            ),
        ).remove_columns(["text"])
    elif dataset_name == "blimp":
        dataset = datasets.load_dataset("WillHeld/blimp", split="train")

        def tokenize_examples(examples):
            good = tokenizer(
                examples["sentence_good"],
                truncation=True,
                max_length=128,
            )
            bad = tokenizer(
                examples["sentence_bad"],
                truncation=True,
                max_length=128,
            )
            return {
                "good_input_ids": good["input_ids"],
                "good_attention_mask": good["attention_mask"],
                "bad_input_ids": bad["input_ids"],
                "bad_attention_mask": bad["attention_mask"],
            }

        cols = dataset.column_names
        dataset = dataset.map(tokenize_examples, batched=True).remove_columns(cols)

    else:
        dataset = load_dataset(
            "Salesforce/wikitext", name="wikitext-2-v1", split="train"
        )
        dataset = dataset.map(
            lambda x: tokenizer(x["text"], truncation=True, max_length=2048),
            batched=True,
        )
    dataset.save_to_disk(out_dir)


if __name__ == "__main__":
    import fire

    fire.Fire(
        {
            "cache_data": cache_data,
            "cache_data_from_hf": cache_data_from_hf,
        }
    )
