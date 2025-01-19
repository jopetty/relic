from sklearn.linear_model import LinearRegression
from scipy.stats import pearsonr

from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
)
import datasets

import torch
import torch.nn.functional as F
import numpy as np
import os
import json
import fire
import glob

from modeling_ppt_neox import PPTNeoXForCausalLM
from utils import load_avg_activations


def id_estimate(X, fraction=0.9, verbose=False):
    """
    From Angie Chen's Sudden Drops in the Loss
    https://openreview.net/forum?id=MO5PiKHELW

    Estimates the intrinsic dimension of a system of points from
    the matrix of their distances X

    Args:
    X : 2-D Matrix X (n,n) where n is the number of points
    fraction : fraction of the data considered for the dimensionality
    estimation (default : fraction = 0.9)
    Returns:
    x : log(mu)    (*)
    y : -(1-F(mu)) (*)
    reg : the intrinsic dimension estimate
    r : determination coefficient of y ~ x
    pval : p-value of y ~ x

    (*) See cited paper for description

    Usage:

    _,_,reg,r,pval = estimate(X,fraction=0.85)

    The technique is described in :

    "Estimating the intrinsic dimension of datasets by a
    minimal neighborhood information"
    Authors : Elena Facco, Maria d’Errico, Alex Rodriguez & Alessandro Laio
    Scientific Reports 7, Article number: 12140 (2017)
    doi:10.1038/s41598-017-11873-y

    """

    # sort distance matrix
    Y = np.sort(X, axis=1, kind="quicksort")

    # clean data
    k1 = Y[:, 1]
    k2 = Y[:, 2]

    zeros = np.where(k1 == 0)[0]
    if verbose:
        print("Found n. {} elements for which r1 = 0".format(zeros.shape[0]))
        print(zeros)

    degeneracies = np.where(k1 == k2)[0]
    if verbose:
        print("Found n. {} elements for which r1 = r2".format(degeneracies.shape[0]))
        print(degeneracies)

    good = np.setdiff1d(np.arange(Y.shape[0]), np.array(zeros))
    good = np.setdiff1d(good, np.array(degeneracies))

    if verbose:
        print("Fraction good points: {}".format(good.shape[0] / Y.shape[0]))

    k1 = k1[good]
    k2 = k2[good]

    # n.of points to consider for the linear regression
    npoints = int(np.floor(good.shape[0] * fraction))

    # define mu and Femp
    N = good.shape[0]
    mu = np.sort(np.divide(k2, k1), axis=None, kind="quicksort")
    Femp = (np.arange(1, N + 1, dtype=np.float64)) / N

    # take logs (leave out the last element because 1-Femp is zero there)
    x = np.log(mu[:-2])
    y = -np.log(1 - Femp[:-2])

    # regression
    x_good = x[0:npoints, np.newaxis]
    y_good = y[0:npoints, np.newaxis]
    not_nan_idx = np.logical_not(np.isnan(x_good))
    x_good = x_good[not_nan_idx][:, np.newaxis]
    y_good = y_good[not_nan_idx][:, np.newaxis]

    regr = LinearRegression(fit_intercept=False)
    regr.fit(x_good, y_good)

    r, pval = pearsonr(x_good.reshape(x_good.shape[0]), y_good.reshape(y_good.shape[0]))
    return x, y, regr.coef_[0][0], r, pval


# def sim_matrix(a, b, eps=1e-8):
#     """
#     Computes pairwise cosine similarity of all rows in a with all rows in b. Eps(ilon) included for numerical stability.
#     """
#     a_n, b_n = a.norm(dim=1)[:, None], b.norm(dim=1)[:, None]
#     a_norm = a / torch.max(a_n, eps * torch.ones_like(a_n))
#     b_norm = b / torch.max(b_n, eps * torch.ones_like(b_n))
#     sim_mt = torch.mm(a_norm, b_norm.transpose(0, 1))
#     return sim_mt


def sim_matrix(a: torch.Tensor, b: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """
    Computes pairwise cosine similarity between all rows in tensors a and b.

    Args:
        a (torch.Tensor): First input tensor of shape (n, d)
        b (torch.Tensor): Second input tensor of shape (m, d)
        eps (float): Small constant for numerical stability

    Returns:
        torch.Tensor: Similarity matrix of shape (n, m) with values in [-1, 1]
    """
    # Ensure inputs are at least 2D
    if a.dim() == 1:
        a = a.unsqueeze(0)
    if b.dim() == 1:
        b = b.unsqueeze(0)

    # Compute L2 norms
    a_squared_norm = torch.sum(a**2, dim=-1, keepdim=True).clamp(min=eps)
    b_squared_norm = torch.sum(b**2, dim=-1, keepdim=True).clamp(min=eps)

    # Compute dot product
    dot_product = torch.matmul(a, b.transpose(-2, -1))

    # Compute similarity matrix
    similarity = dot_product / torch.sqrt(
        a_squared_norm * b_squared_norm.transpose(-2, -1)
    )

    # Clip values to [-1, 1] range to handle numerical instabilities
    similarity = torch.clamp(similarity, min=-1.0, max=1.0)

    return similarity


def convert_np_to_py_type(x):
    if type(x).__module__ == "numpy":
        return x.item()
    else:
        return x


def compute_stats(
    model_dir,
    compute_loss=True,
    compute_weight_norm=True,
    compute_id=True,
    sample_size=1024,
    bsz=1,
    load_pruned_model=False,
    log_alpha_path=None,
):
    if log_alpha_path is not None:
        log_alpha_name = log_alpha_path.split("/")[-1].split(".")[0]
    else:
        log_alpha_name = None
    id_output_path = os.path.join(
        model_dir, f"id_{load_pruned_model}_{log_alpha_name}.json"
    )
    if os.path.exists(id_output_path):
        print(f"{id_output_path} already exists.")
        compute_id = False

    weight_norm_output_path = os.path.join(
        model_dir, f"weight_norm_{load_pruned_model}_{log_alpha_name}.json"
    )
    if os.path.exists(weight_norm_output_path):
        print(f"{weight_norm_output_path} already exists.")
        compute_weight_norm = False

    output_bool_vals = {
        "loss": compute_loss,
        "intrinsic dimension": compute_id,
        "weight_norms": compute_weight_norm,
    }
    if not compute_loss and not compute_id and not compute_weight_norm:
        print("Not computing loss, intrinsic dimension, or weight norm. Returning...")
        return
    else:
        print(f"Computing {', '.join([k for k, v in output_bool_vals.items() if v])}.")

    dataset = (
        datasets.load_dataset(
            "json",
            data_files=[
                f"/vast/work/public/ml-datasets/c4/en/c4-validation.0000{i}-of-00008.json"
                for i in range(8)
            ],
        )["train"]
        # .shuffle(0)
        .select(range(sample_size))
    )

    if load_pruned_model:
        model = PPTNeoXForCausalLM.from_pretrained(model_dir).cuda().eval()
        log_alpha_heads = torch.tensor(
            np.load(log_alpha_path), dtype=torch.float32
        ).cuda()
        model.gpt_neox.set_log_alpha_params(log_alpha_heads)

        avg_activation_path = os.path.join(model_dir, "avg_activations.pkl")
        load_avg_activations(model, avg_activation_path, "cuda")
    else:
        model = AutoModelForCausalLM.from_pretrained(model_dir).cuda().eval()

    tokenizer = AutoTokenizer.from_pretrained("EleutherAI/pythia-160m")
    tokenizer.add_special_tokens({"pad_token": "<|padding|>"})

    def tokenize_examples(examples):
        return tokenizer(
            examples["text"],
            padding="max_length",
            truncation=True,
            max_length=2048,
        )

    def get_lengths_and_labels(examples):
        input_ids = examples["input_ids"]
        attention_mask = examples["attention_mask"]

        labels = []
        lengths = []

        for i in range(len(input_ids)):  # Iterate through the batch
            current_input_ids = input_ids[i]
            current_attention_mask = attention_mask[i]

            # 1. Set pad tokens to -100 in labels (using list comprehension)
            current_labels = [
                -100 if token == tokenizer.pad_token_id else token
                for token in current_input_ids
            ]
            labels.append(current_labels)

            # 2. Get the length of the sequence before padding
            length = sum(current_attention_mask) - 1
            lengths.append(length)

        examples["labels"] = labels
        examples["lengths"] = lengths
        return examples

    collated = dataset.map(tokenize_examples, batched=True).map(
        get_lengths_and_labels, batched=True
    )

    model_kwargs = {
        "return_dict": True,
    }
    cls_embeddings = []
    if compute_id:
        model_kwargs["output_hidden_states"] = True

    @torch.inference_mode()
    def compute_outputs(
        model, model_kwargs, input_ids, attention_mask, labels, lengths
    ):
        output_dict = {}
        lengths = torch.tensor(lengths).cuda()
        if compute_loss:
            model_kwargs["labels"] = torch.tensor(labels).cuda()

        outputs = model(
            input_ids=torch.tensor(input_ids).cuda(),
            attention_mask=torch.tensor(attention_mask).cuda(),
            **model_kwargs,
        )

        if compute_weight_norm:
            param_weights = []
            for name, param in model.named_parameters():
                if compute_weight_norm and param.data is not None:
                    param_weights.append(param.data.cpu().detach().numpy().flatten())
            if compute_weight_norm:
                output_dict["weight_norms"] = [
                    (np.linalg.norm(np.concatenate(param_weights)) ** 2).item()
                ]

        if compute_loss:
            if isinstance(outputs, dict):
                output_dict["loss"] = [outputs["loss"].item()]
            else:
                output_dict["loss"] = [outputs.loss.item()]
        if compute_id:
            if isinstance(outputs, dict):
                for layer in range(12):
                    output_dict[f"cls_embeddings_{layer}"] = [
                        outputs["hidden_states"][layer][
                            torch.arange(len(lengths)), lengths, :
                        ]
                    ]
            else:
                for layer in range(12):
                    output_dict[f"cls_embeddings_{layer}"] = [
                        outputs.hidden_states[layer][
                            torch.arange(len(lengths)), lengths, :
                        ]
                    ]
        return output_dict

    cols = collated.column_names
    collated = collated.map(
        lambda ex: compute_outputs(
            model,
            model_kwargs,
            ex["input_ids"],
            ex["attention_mask"],
            ex["labels"],
            ex["lengths"],
        ),
        batched=True,
        batch_size=bsz,
        remove_columns=cols,
    )

    if compute_id:
        ret = {}
        for layer in range(12):
            cls_embeddings = list(
                [torch.FloatTensor(x) for x in collated[f"cls_embeddings_{layer}"]]
            )
            # Calculate cosine distances and use pairwise distances to estimate dataset ID
            # Calculate on first 1000
            cls_embeddings = torch.cat(cls_embeddings, dim=0)
            emb_sim_matrix = (
                sim_matrix(cls_embeddings, cls_embeddings).cpu().detach().numpy()
            )
            _, _, d, r, pval = id_estimate(
                1 - emb_sim_matrix, fraction=1.0, verbose=False
            )
            print(
                f"Layer: {layer}. Intrinsic dimension: {d}, Pearson corr coeff: {r}, p-value: {pval}"
            )
            ret[layer] = {
                "intrinsic_dimension": convert_np_to_py_type(d),
                "pearson_corr_coeff": convert_np_to_py_type(r),
                "p-value": convert_np_to_py_type(pval),
            }

        with open(id_output_path, "w") as f:
            json.dump(ret, f)

    if compute_loss:
        # write out loss
        loss = list(collated["loss"])
        with open(
            os.path.join(model_dir, f"loss_{load_pruned_model}_{log_alpha_name}.json"),
            "w",
        ) as f:
            f.write(json.dumps(loss) + "\n")

    if compute_weight_norm:
        # weight_norms = list(collated["weight_norms"])
        param_weights = []
        for _, param in model.named_parameters():
            param_weights.append(param.data.cpu().detach().numpy().flatten())
        weight_norms = (np.linalg.norm(np.concatenate(param_weights)) ** 2).item()
        with open(weight_norm_output_path, "w") as f:
            f.write(json.dumps(weight_norms) + "\n")


def blimp_eval(model_dir, bsz=128, load_pruned_model=False, log_alpha_path=None):
    if log_alpha_path is not None:
        log_alpha_name = log_alpha_path.split("/")[-1].split(".")[0]
    else:
        log_alpha_name = None

    if os.path.exists(
        os.path.join(model_dir, f"blimp_{load_pruned_model}_{log_alpha_path}.json")
    ):
        print(f"{model_dir} blimp already exists.")
        return

    dataset = datasets.load_dataset("WillHeld/blimp")["train"]
    tokenizer = AutoTokenizer.from_pretrained("EleutherAI/pythia-160m")
    tokenizer.add_special_tokens({"pad_token": "<|padding|>"})

    if load_pruned_model:
        model = PPTNeoXForCausalLM.from_pretrained(model_dir).cuda().eval()
        log_alpha_heads = torch.tensor(
            np.load(log_alpha_path), dtype=torch.float32
        ).cuda()
        model.gpt_neox.set_log_alpha_params(log_alpha_heads)

        avg_activation_path = os.path.join(model_dir, "avg_activations_blimp.pkl")
        load_avg_activations(model, avg_activation_path, "cuda")
    else:
        model = AutoModelForCausalLM.from_pretrained(model_dir).cuda()

    # tokenize sentence_good and sentence_bad
    def tokenize_examples(examples):
        good = tokenizer(
            examples["sentence_good"],
            padding="max_length",
            truncation=True,
            max_length=128,
        )
        bad = tokenizer(
            examples["sentence_bad"],
            padding="max_length",
            truncation=True,
            max_length=128,
        )
        return {
            "good_input_ids": good["input_ids"],
            "good_attention_mask": good["attention_mask"],
            "bad_input_ids": bad["input_ids"],
            "bad_attention_mask": bad["attention_mask"],
        }

    tokenized = dataset.map(tokenize_examples, batched=True)  # .select(range(4096))

    model_kwargs = {
        "return_dict": True,
    }

    @torch.inference_mode()
    def compute_batch_accuracy(
        model,
        model_kwargs,
        good_input_ids,
        good_attention_mask,
        bad_input_ids,
        bad_attention_mask,
    ):
        good_input_ids = torch.tensor(good_input_ids).cuda()
        good_attention_mask = torch.tensor(good_attention_mask).cuda()
        bad_input_ids = torch.tensor(bad_input_ids).cuda()
        bad_attention_mask = torch.tensor(bad_attention_mask).cuda()
        good_output = model(
            input_ids=good_input_ids,
            attention_mask=good_attention_mask,
            **model_kwargs,
        )
        bad_output = model(
            input_ids=bad_input_ids,
            attention_mask=bad_attention_mask,
            **model_kwargs,
        )
        good_ll = torch.tensor(
            [
                -F.cross_entropy(
                    good_output.logits[i, : length - 1],
                    good_input_ids[i, 1:length],
                    reduction="sum",
                )
                for i, length in enumerate(
                    (good_input_ids != tokenizer.pad_token_id).sum(1)
                )
            ]
        )

        bad_ll = torch.tensor(
            [
                -F.cross_entropy(
                    bad_output.logits[i, : length - 1],
                    bad_input_ids[i, 1:length],
                    reduction="sum",
                )
                for i, length in enumerate(
                    (bad_input_ids != tokenizer.pad_token_id).sum(1)
                )
            ]
        )
        is_correct = good_ll > bad_ll
        return {"correct": is_correct.cpu().numpy()}

    cols = tokenized.column_names
    tokenized = tokenized.map(
        lambda ex: compute_batch_accuracy(
            model,
            model_kwargs,
            ex["good_input_ids"],
            ex["good_attention_mask"],
            ex["bad_input_ids"],
            ex["bad_attention_mask"],
        ),
        batched=True,
        batch_size=bsz,
        remove_columns=cols,
    )
    correct = np.sum(list(tokenized["correct"])) / 67000
    correct = convert_np_to_py_type(correct)

    with open(
        os.path.join(model_dir, f"blimp_{load_pruned_model}_{log_alpha_name}.json"), "w"
    ) as f:
        f.write(json.dumps(correct) + "\n")


def main(super_dir, bsz=32, sample_size=32):
    for model_dir in glob.glob(super_dir + "/*"):
        print(f"Processing {model_dir}")
        compute_stats(model_dir, bsz=bsz, sample_size=sample_size)


if __name__ == "__main__":
    fire.Fire({"main": main, "blimp": blimp_eval, "compute_stats": compute_stats})
