import datasets
from datasets import load_dataset, load_from_disk, Dataset
from collections import defaultdict
import os
from transformers import AutoTokenizer
import random
import fire
from tqdm import tqdm, trange


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
    elif dataset_name == "wikitext":
        dataset = load_dataset(
            "Salesforce/wikitext", name="wikitext-2-v1", split="train"
        )
        dataset = dataset.map(
            lambda x: tokenizer(x["text"], truncation=True, max_length=2048),
            batched=True,
        )
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
        dataset = load_dataset("text", data_files=dataset_name, split="train")
        dataset = dataset.map(
            lambda x: tokenizer(
                x["text"],
                truncation=True,
                max_length=2048,
            ),
        ).remove_columns(["text"])

    dataset.save_to_disk(out_dir)


def train_ngram_models(dataset):
    """
    Trains unigram, bigram, and trigram language models on a tokenized dataset.
    Modifies unigram model to exclude <s> from being generated after the start.

    Args:
      dataset: A tokenized Hugging Face dataset.

    Returns:
      A tuple of three dictionaries representing the unigram, bigram, and trigram models.
    """

    unigram_model = defaultdict(int)
    bigram_model = defaultdict(lambda: defaultdict(int))
    trigram_model = defaultdict(lambda: defaultdict(int))

    for example in tqdm(dataset):
        tokens = example["input_ids"]
        tokens = ["<s>"] + tokens  # Add start token

        for i in range(len(tokens)):
            unigram_model[tuple(tokens[i : i + 1])] += 1  # Unigram
            if i < len(tokens) - 1:
                bigram_model[tuple(tokens[i : i + 1])][tokens[i + 1]] += 1  # Bigram
            if i < len(tokens) - 2:
                trigram_model[tuple(tokens[i : i + 2])][tokens[i + 2]] += 1  # Trigram

    # --- Modification to prevent <s> generation after start ---
    unigram_model_no_s = unigram_model.copy()
    del unigram_model_no_s[("<s>",)]  # Remove <s> from unigram model

    # Normalize counts to probabilities

    # Unigram Normalization (for generation without <s>)
    total_unigram_count = sum(unigram_model_no_s.values())
    for unigram in unigram_model_no_s:
        unigram_model_no_s[unigram] /= total_unigram_count

    # Bigram and Trigram Normalization
    for model in [bigram_model, trigram_model]:
        for prev_words in model:
            total_count = sum(model[prev_words].values())
            for word in model[prev_words]:
                model[prev_words][word] /= total_count

    # Return the modified unigram model without <s>
    return unigram_model_no_s, bigram_model, trigram_model


def generate_text(
    unigram_model,
    bigram_model,
    trigram_model,
    use_trigram=False,
    use_bigram=False,
    length=20,
):
    """
    Generates text using a trigram model with backoff to bigram and unigram models.
    Uses sampling instead of argmax.

    Args:
      unigram_model: The trained unigram model.
      bigram_model: The trained bigram model.
      trigram_model: The trained trigram model.
      use_trigram: Whether to use the trigram model if available.
      use_bigram: Whether to use the bigram model if available.
      length: The desired length of the generated text (in tokens).

    Returns:
      A list containing the generated tokens (excluding the start token).
    """

    text = ["<s>"]

    for _ in range(length):
        if len(text) >= 2 and tuple(text[-2:]) in trigram_model and use_trigram:
            # Trigram sampling
            candidates = list(trigram_model[tuple(text[-2:])].keys())
            probabilities = list(trigram_model[tuple(text[-2:])].values())
            next_word = random.choices(candidates, probabilities)[0]
        elif len(text) >= 1 and tuple(text[-1:]) in bigram_model and use_bigram:
            # Bigram sampling
            candidates = list(bigram_model[tuple(text[-1:])].keys())
            probabilities = list(bigram_model[tuple(text[-1:])].values())
            next_word = random.choices(candidates, probabilities)[0]
        else:
            # Unigram sampling
            candidates = list(unigram_model.keys())
            probabilities = list(unigram_model.values())
            next_word = random.choices(candidates, probabilities)[0][0]

        text.append(next_word)

    return text[1:]


def make_dummy_tasks(data_dir, out_dir, length=2048):
    dataset = load_from_disk(data_dir)
    unigram_model, bigram_model, trigram_model = train_ngram_models(dataset)

    # Generate text for each row and store in a list
    new_unigram = []
    new_bigram = []
    new_trigram = []

    for i in trange(len(dataset)):
        text = generate_text(
            unigram_model,
            bigram_model,
            trigram_model,
            use_trigram=True,
            use_bigram=True,
            length=2048,
        )
        new_trigram.append({"input_ids": text})

        text = generate_text(
            unigram_model,
            bigram_model,
            trigram_model,
            use_trigram=False,
            use_bigram=True,
            length=2048,
        )
        new_bigram.append({"input_ids": text})

        text = generate_text(
            unigram_model,
            bigram_model,
            trigram_model,
            use_trigram=False,
            use_bigram=False,
            length=2048,
        )
        new_unigram.append({"input_ids": text})

    unigram_dataset = Dataset.from_list(new_unigram)
    bigram_dataset = Dataset.from_list(new_bigram)
    trigram_dataset = Dataset.from_list(new_trigram)

    unigram_dataset.save_to_disk(os.path.join(out_dir, "unigram"))
    bigram_dataset.save_to_disk(os.path.join(out_dir, "bigram"))
    trigram_dataset.save_to_disk(os.path.join(out_dir, "trigram"))


def shuffle_spans(dataset, k, deterministic=True):
    """
    Shuffles spans of k tokens in a Hugging Face dataset.
    Allows toggling between deterministic and nondeterministic shuffling.

    Args:
      dataset: A tokenized Hugging Face dataset.
      k: The length of the spans to shuffle.
      deterministic: If True, uses a fixed seed for shuffling, resulting in the same
        shuffled dataset every time. If False, each span is shuffled randomly.

    Returns:
      A new Hugging Face dataset with shuffled spans.
    """

    # Generate shuffle order at the beginning only if deterministic
    if deterministic:
        shuffle_order = list(range(k))
        random.Random(4).shuffle(
            shuffle_order
        )  # Fixed seed for deterministic shuffling

    def shuffle_tokens(examples):
        new_tokens = []
        for example in examples["input_ids"]:
            tokens = example
            shuffled_example = []
            for i in range(0, len(tokens), k):
                span = tokens[i : i + k]
                if len(span) == k:
                    if deterministic:
                        # Use the pre-generated shuffle order
                        current_shuffle_order = shuffle_order
                    else:
                        # Generate a random shuffle order for this span
                        current_shuffle_order = list(range(k))
                        random.shuffle(current_shuffle_order)

                    shuffled_span = [span[j] for j in current_shuffle_order]
                    shuffled_example.extend(shuffled_span)
                else:
                    shuffled_example.extend(span)
            new_tokens.append(shuffled_example)
        return {"input_ids": new_tokens}

    shuffled_dataset = dataset.map(shuffle_tokens, batched=True)
    return shuffled_dataset


def full_shuffle(dataset):
    """
    Nondeterministically shuffles the tokens in a Hugging Face dataset.
    """

    def shuffle_tokens(examples):
        new_tokens = []
        for example in examples["input_ids"]:
            shuffled_example = example[:]  # Create a copy to avoid modifying in place
            random.shuffle(shuffled_example)
            new_tokens.append(shuffled_example)
        return {"input_ids": new_tokens}  # Return modified tokens

    shuffled_dataset = dataset.map(shuffle_tokens, batched=True)
    return shuffled_dataset


def make_shuffle_tasks(data_dir, out_dir):
    dataset = load_from_disk(data_dir)

    # Shuffle spans of 4 tokens
    shuffled_dataset = shuffle_spans(dataset, 4, deterministic=False)
    shuffled_dataset.save_to_disk(os.path.join(out_dir, "shuffle_spans_4_nd"))

    # Shuffle spans of 8 tokens
    shuffled_dataset = shuffle_spans(dataset, 8, deterministic=False)
    shuffled_dataset.save_to_disk(os.path.join(out_dir, "shuffle_spans_8_nd"))

    # Full shuffle
    full_shuffled_dataset = full_shuffle(dataset)
    full_shuffled_dataset.save_to_disk(os.path.join(out_dir, "full_shuffle"))


def make_random_tasks(out_dir, tokenizer_name="EleutherAI/pythia-160m"):
    """
    Generates random binary strings and random integer strings of length 0-60.
    Tokenizes them and saves them as Hugging Face datasets.
    """
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    tokenizer.add_special_tokens({"pad_token": "<|padding|>"})

    def random_binary_string():
        return " ".join(random.choice("01") for _ in range(2048))

    def random_int_string_60():
        return " ".join(random.choice("0123456789") for _ in range(2048))  # Only digits

    num_tasks = 100000  # 100k examples per dataset

    binary_tasks = [{"text": random_binary_string()} for _ in range(num_tasks)]
    int_string_tasks = [{"text": random_int_string_60()} for _ in range(num_tasks)]

    def tokenize_function(examples):
        return tokenizer(examples["text"], padding="max_length", truncation=True)

    # Create and tokenize datasets
    binary_dataset = Dataset.from_list(binary_tasks).map(
        tokenize_function, batched=True
    )
    int_string_dataset = Dataset.from_list(int_string_tasks).map(
        tokenize_function, batched=True
    )

    # Ensure output directory exists
    os.makedirs(out_dir, exist_ok=True)

    # Save datasets to disk
    binary_dataset.save_to_disk(os.path.join(out_dir, "random_binary_dataset"))
    int_string_dataset.save_to_disk(os.path.join(out_dir, "random_int_string_dataset"))

    print(f"Generated and saved datasets in {out_dir}")


if __name__ == "__main__":
    fire.Fire(
        {
            "cache_data": cache_data,
            "cache_data_from_hf": cache_data_from_hf,
            "make_dummy_tasks": make_dummy_tasks,
            "make_shuffle_tasks": make_shuffle_tasks,
            "make_random_tasks": make_random_tasks,
        }
    )
