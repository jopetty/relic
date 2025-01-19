import math
import torch
import os
from typing import Dict, Union, Any
import pickle
from copy import deepcopy
from torch.optim import AdamW

import fire
import datasets
import wandb

from transformers import (
    Trainer,
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    get_linear_schedule_with_warmup,
    DataCollatorForLanguageModeling,
)
from modeling_ppt_neox import PPTNeoXForCausalLM
from utils import freeze_all_expecting_pruning_params, load_avg_activations


def get_optimizers(model, lr, reg_lr, num_training_steps, warmup_steps=0):
    optimizer_1_group = []
    optimizer_2_group = []

    for n, p in model.named_parameters():
        if "log_alpha" in n:
            optimizer_1_group.append(p)
        elif "sparsity_lambda" in n:
            optimizer_2_group.append(p)

    optimizer = AdamW(
        [
            {
                "params": optimizer_1_group,
                "maximize": False,  # The log alphas try to minimize the loss
                "lr": lr,
            },
            {
                "params": optimizer_2_group,
                "maximize": True,  # The regularization lambdas try to maximize the penalty
                "lr": reg_lr,
            },
        ],
    )
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=warmup_steps, num_training_steps=num_training_steps
    )

    return optimizer, scheduler


class Pruner(Trainer):
    def __init__(self, *args, **kwargs):
        self.target_sparsity = kwargs.pop("target_sparsity", 0.0)
        self.start_sparsity = kwargs.pop("start_sparsity", 0.0)
        self.num_sparsity_warmup_steps = kwargs.pop("num_sparsity_warmup_steps", 0)
        self.warmup_type = kwargs.pop("warmup_type", "linear")
        self.ref_model = kwargs.pop("ref_model", None)
        self.reg_lr = kwargs.pop("reg_lr", 0.1)  # Add reg_lr as a class parameter
        super().__init__(*args, **kwargs)

        # Initialize lambda clips
        self.lambda_min = 0.0
        self.lambda_max = float("inf")  # Or set a specific upper bound if desired

    def get_current_target_sparsity(self, global_step):
        if global_step < self.num_sparsity_warmup_steps:
            if self.warmup_type == "linear":
                return (
                    self.start_sparsity
                    + (self.target_sparsity - self.start_sparsity)
                    * global_step
                    / self.num_sparsity_warmup_steps
                )
            else:
                raise ValueError(f"Unknown warmup type: {self.warmup_type}")
        else:
            return self.target_sparsity

    def compute_loss(self, model, inputs, num_items_in_batch, return_outputs=False):
        outputs = model(
            **inputs,
            target_sparsity=self.get_current_target_sparsity(self.state.global_step),
        )

        zs_loss = outputs.zs_loss
        logits = outputs.logits

        with torch.inference_mode():
            ref_logits = self.ref_model(**inputs).logits

        logits = torch.nn.functional.log_softmax(logits, dim=-1)
        ref_logits = torch.nn.functional.log_softmax(ref_logits, dim=-1)

        kl_loss = torch.nn.functional.kl_div(
            logits, ref_logits, reduction="batchmean", log_target=True
        )

        loss = zs_loss + kl_loss

        current_sparsity = 1 - outputs.z_sum / model.num_alpha_params
        target_sparsity = self.get_current_target_sparsity(self.state.global_step)

        wandb.log(
            {
                "sparsity": current_sparsity,
                "target_sparsity": target_sparsity,
                "zs_loss": zs_loss,
                "kl_loss": kl_loss,
            },
            step=self.state.global_step,
        )

        return (loss, outputs) if return_outputs else loss


def main(
    data_dir="./data/tokenized/depth9_train",
    model_name="EleutherAI/pythia-160m",
    gradient_accumulation_steps=1,
    max_steps=5000,
    bsz=8,
    warmup_steps=500,
    logging_steps=1,
    save_steps=125,
    output_dir="output",
    seed=3407,
    report_to="wandb",
    lr=0.1,
    reg_lr=1,
    target_sparsity=0.5,
    sparsity_warmup_steps=1000,
):
    print(locals())

    dataset = datasets.load_from_disk(data_dir)

    if "train" in dataset:
        dataset = dataset["train"]

    model = (
        PPTNeoXForCausalLM.from_pretrained(model_name, attn_implementation="eager")
    ).cuda()
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    ref_model = AutoModelForCausalLM.from_pretrained(model_name).cuda().eval()

    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer, mlm=False, pad_to_multiple_of=2048
    )

    avg_activation_path = os.path.join(model_name, "avg_activations.pkl")
    load_avg_activations(model, avg_activation_path, "cuda")
    freeze_all_expecting_pruning_params(model)

    optimizers = get_optimizers(
        model,
        lr=lr,
        reg_lr=reg_lr,
        num_training_steps=max_steps,
        warmup_steps=warmup_steps,
    )

    training_args = TrainingArguments(
        per_device_train_batch_size=bsz,
        gradient_accumulation_steps=gradient_accumulation_steps,
        warmup_steps=warmup_steps,
        max_steps=max_steps,
        logging_steps=logging_steps,
        save_strategy="steps",
        save_steps=save_steps,
        output_dir=output_dir,
        seed=seed,
        report_to=report_to,
    )

    trainer = Pruner(
        model=model,
        ref_model=ref_model,
        args=training_args,
        train_dataset=dataset,
        optimizers=optimizers,
        data_collator=data_collator,
        target_sparsity=target_sparsity,
        num_sparsity_warmup_steps=sparsity_warmup_steps,
    )

    trainer.train()


if __name__ == "__main__":
    fire.Fire(main)
