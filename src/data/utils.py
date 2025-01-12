import datasets
from datasets import load_dataset, IterableDataset, load_from_disk
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
    # tokenizer.add_special_tokens({"pad_token": "<|padding|>"})

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
    else:
        dataset = load_dataset(
            "Salesforce/wikitext", name="wikitext-2-v1", split="train"
        )
        dataset = dataset.map(
            lambda x: tokenizer(x["text"], truncation=True, max_length=2048),
            batched=True,
        )
    dataset.save_to_disk(out_dir)


def domain_gen(data, seq_len, domain_id=None):
    random_order = np.random.permutation(len(data) // seq_len)
    if domain_id is None:
        for i in random_order:
            yield {"input_ids": data[i * seq_len : (i + 1) * seq_len]}
    else:
        for i in random_order:
            yield {
                "domain_id": torch.tensor([domain_id], dtype=torch.long),
                "input_ids": data[i * seq_len : (i + 1) * seq_len],
            }


def domain_gen_train(data, seq_len, domain_id=None):
    data_len = len(data)
    num_sequences = data_len // seq_len

    while True:
        random_order = np.random.permutation(num_sequences)
        for i in random_order:
            if domain_id is None:
                yield {
                    "input_ids": torch.tensor(
                        data[i * seq_len : (i + 1) * seq_len].astype(np.int64)
                    )
                }
            else:
                yield {
                    "domain_id": torch.tensor([domain_id], dtype=torch.long),
                    "input_ids": torch.tensor(
                        data[i * seq_len : (i + 1) * seq_len].astype(np.int64)
                    ),
                }


# def get_dataset(dataset_name, **dataset_kwargs):
#     if dataset_name == "slimpj":
#         return get_slimpj_dataset(**dataset_kwargs)
#     else
#         return datasets.load_from_disk(DATA_PATHS[dataset_name])


def get_slimpj_dataset(seed, is_eval, seq_len):
    if is_eval:
        data = np.fromfile(
            os.path.join(REDPAJAMA_DATA_PATH, "val.bin"), dtype=np.uint16
        )
        return IterableDataset.from_generator(
            domain_gen, gen_kwargs={"data": data, "seq_len": seq_len}
        )
    else:
        data = np.fromfile(
            os.path.join(REDPAJAMA_DATA_PATH, "train.bin"), dtype=np.uint16
        )
        return IterableDataset.from_generator(
            domain_gen_train, gen_kwargs={"data": data, "seq_len": seq_len}
        )


def get_slimpajama_6b(
    tokenizer_name="EleutherAI/pythia-160m", num_proc=16, return_torch=False
):
    """Full: https://huggingface.co/datasets/cerebras/SlimPajama-627B
    6B-subset: DKYoon/SlimPajama-6B
    """
    # {
    #     "text": ...,
    #     "meta": {"url": "...", "timestamp": "...", "source": "...", "language": "...", ...},
    #     "red_pajama_subset": "common_crawl" | "c4" | "github" | "books" | "arxiv" | "wikipedia" | "stackexchange"
    # }

    if tokenizer_name is None:
        raise ValueError("Please specify a tokenizer name.")

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    if "pythia" in tokenizer_name:
        REDPAJAMA_DATA_PATH = os.path.join(
            "/scratch/myh2014/data/", f"datasets/slim_6b_pythia/"
        )
    else:
        REDPAJAMA_DATA_PATH = os.path.join(
            "/scratch/myh2014/data/", f"datasets/slim_6b_{tokenizer_name}/"
        )

    if not os.path.exists(os.path.join(REDPAJAMA_DATA_PATH, "val.bin")):
        os.makedirs(REDPAJAMA_DATA_PATH, exist_ok=True)
        dataset = load_dataset(
            "DKYoon/SlimPajama-6B",
            split=["train", "test"],
            cache_dir="/scratch/myh2014/data/",
        )

        def process_hf_tokenizer(example):
            "Processing dataset..."
            ids = tokenizer(example["text"])[
                "input_ids"
            ]  # encode_ordinary ignores any special tokens
            ids.append(
                tokenizer.eos_token_id
            )  # add the end of text token, e.g. 50256 for gpt2 bpe
            out = {"ids": ids, "len": len(ids)}
            return out

        # tokenize the dataset
        tokenized = {}

        tokenized["train"] = dataset[0].map(
            process_hf_tokenizer,
            remove_columns=["text", "meta", "__index_level_0__"],
            desc="tokenizing the splits",
            num_proc=num_proc,
        )
        tokenized["val"] = dataset[1].map(
            process_hf_tokenizer,
            remove_columns=["text", "meta", "__index_level_0__"],
            desc="tokenizing the splits",
            num_proc=num_proc,
        )

        # concatenate all the ids in each dataset into one large file we can use for training
        for split, dset in tokenized.items():
            print("Columns: ", dset.features)
            arr_len = np.sum(dset["len"])
            filename = os.path.join(REDPAJAMA_DATA_PATH, f"{split}.bin")
            dtype = np.uint16  # (can do since enc.max_token_value == 50256 is < 2**16)
            arr = np.memmap(filename, dtype=dtype, mode="w+", shape=(arr_len,))
            total_batches = 10

            idx = 0
            for batch_idx in tqdm(range(total_batches), desc=f"writing {filename}"):
                # Batch together samples for faster write
                batch = dset.shard(
                    num_shards=total_batches, index=batch_idx, contiguous=True
                ).with_format("numpy")
                arr_batch = np.concatenate(batch["ids"])
                # Write into mmap
                arr[idx : idx + len(arr_batch)] = arr_batch
                idx += len(arr_batch)
            arr.flush()

    train_data = np.memmap(
        os.path.join(REDPAJAMA_DATA_PATH, "train.bin"), dtype=np.uint16, mode="r"
    )
    val_data = np.memmap(
        os.path.join(REDPAJAMA_DATA_PATH, "val.bin"), dtype=np.uint16, mode="r"
    )
    if return_torch:
        train_data = torch.tensor(np.array(train_data, dtype=np.int32))
        val_data = torch.tensor(np.array(val_data, dtype=np.int32))

    return {"train": train_data, "val": val_data}


if __name__ == "__main__":
    import fire

    fire.Fire(
        {
            "cache_data": cache_data,
            "cache_data_from_hf": cache_data_from_hf,
            "get_slimpajama_6b": get_slimpajama_6b,
        }
    )
