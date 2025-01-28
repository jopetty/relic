from unsloth import is_bfloat16_supported
from typing import List

from transformers import (
    AutoModelForCausalLM,
    AutoConfig,
    AutoTokenizer,
    set_seed,
    TrainerCallback,
)
from trl import SFTConfig, SFTTrainer
import torch
import datasets
import fire


class SaveAtStepsCallback(TrainerCallback):
    """Custom callback to save model at specific training steps."""

    def __init__(self, save_steps: List[int], output_dir: str):
        """
        Args:
            save_steps: List of steps at which to save the model
            output_dir: Base directory for saving checkpoints
        """
        self.save_steps = sorted(save_steps)  # Sort steps in ascending order
        self.output_dir = output_dir

    def on_step_end(self, args, state, control, **kwargs):
        """Called at the end of each step."""
        if state.global_step in self.save_steps:
            # Create a subdirectory for this specific step
            checkpoint_dir = f"{self.output_dir}/checkpoint-{state.global_step}"
            kwargs["model"].save_pretrained(checkpoint_dir)

            # If you're using a tokenizer, you might want to save it too
            if "tokenizer" in kwargs:
                kwargs["tokenizer"].save_pretrained(checkpoint_dir)

            print(f"Saved model at step {state.global_step}")


def main(
    data_dir="./data/tokenized/depth9_train",
    model_name="EleutherAI/pythia-160m",
    # revision="main",
    reinit=False,
    max_seq_length=2048,
    gradient_accumulation_steps=2,
    max_steps=5000,
    bsz=16,
    warmup_steps=500,
    logging_steps=1,
    save_steps=125,
    output_dir="output",
    # optim="adamw_8bit",
    seed=3407,
    report_to="wandb",
    lr=5e-5,
    min_lr_rate=0.1,
    override_packing=False,
    use_callback=False,
):
    print(locals())
    set_seed(seed)

    callback = SaveAtStepsCallback(
        save_steps=list(range(0, 4000, 100)) + list(range(4000, 10000, 1000)),
        output_dir=output_dir,
    )

    is_a100 = is_bfloat16_supported()

    if data_dir == "c4":
        dataset = datasets.load_dataset(
            "json",
            data_files=[
                f"/vast/work/public/ml-datasets/c4/en/c4-train.0000{i}-of-01024.json"
                for i in range(4)
            ],
        )
    elif data_dir == "c4_2":
        dataset = datasets.load_dataset(
            "json",
            data_files=[
                f"/vast/work/public/ml-datasets/c4/en/c4-train.0000{i}-of-01024.json"
                for i in range(4, 8)
            ],
        )
    elif data_dir == "babylm":
        dataset = datasets.load_dataset("ltg/babylm-2024-baby-cosmo-fine-100m")
    else:
        dataset = datasets.load_from_disk(data_dir)

    if "train" in dataset:
        dataset = dataset["train"]

    if reinit:
        config = AutoConfig.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_config(
            config,
            torch_dtype=torch.bfloat16 if is_a100 else torch.float32,
        )
    else:
        # on a100s, does unsloth pick the right attn implementation?
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            # revision=revision,
            torch_dtype=torch.bfloat16 if is_a100 else torch.float32,
            trust_remote_code=True,
        )
    model.cuda()

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.add_special_tokens({"pad_token": "<|padding|>"})

    packing = any(
        # the datasets that are not pre-packed to 2048
        keyword in data_dir
        for keyword in ["code", "cogs", "arithmetic", "s5", "c4", "babylm"]
    )

    training_args = SFTConfig(
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
        learning_rate=lr,
        lr_scheduler_type="cosine_with_min_lr",
        lr_scheduler_kwargs={"min_lr_rate": min_lr_rate},
        packing=packing if not override_packing else False,
    )

    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        dataset_text_field="text",
        tokenizer=tokenizer,
        max_seq_length=max_seq_length,
    )
    if use_callback:
        trainer.add_callback(callback)

    trainer.train()


if __name__ == "__main__":
    fire.Fire(main)
