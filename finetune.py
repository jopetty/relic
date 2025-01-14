from unsloth import is_bfloat16_supported

from transformers import (
    AutoModelForCausalLM,
    AutoConfig,
    AutoTokenizer,
    DataCollatorWithFlattening,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
)
from trl import SFTConfig, SFTTrainer
import torch
import datasets
import fire

from src.data.utils import get_slimpj_dataset


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
):
    print(locals())

    dataset = datasets.load_from_disk(data_dir)

    if "train" in dataset:
        dataset = dataset["train"]

    if reinit:
        config = AutoConfig.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_config(
            config,
            torch_dtype=torch.bfloat16 if is_bfloat16_supported() else torch.float32,
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            # revision=revision,
            # attn_implementation="flash_attention_2" if is_bfloat16_supported() else "sdpa",
            torch_dtype=torch.bfloat16 if is_bfloat16_supported() else torch.float32,
            trust_remote_code=True,
        )
    model.cuda()

    packing = any(
        # the datasets that are not pre-packed to 2048
        keyword in data_dir
        for keyword in ["code", "cogs", "arithmetic", "s5"]
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
        packing=packing,
    )

    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
    )

    trainer.train()


if __name__ == "__main__":
    fire.Fire(main)
