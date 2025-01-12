from unsloth import is_bfloat16_supported

from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorWithFlattening,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
)
import torch
import datasets
import fire

from src.data.utils import get_slimpj_dataset


def main(
    data_dir="./data/tokenized/depth9_train",
    model_name="EleutherAI/pythia-160m",
    # revision="main",
    max_seq_length=2048,
    gradient_accumulation_steps=2,
    max_steps=5000,
    bsz=16,
    warmup_steps=500,
    logging_steps=1,
    save_steps=125,
    output_dir="outputs",
    # optim="adamw_8bit",
    seed=3407,
    report_to="wandb",
    lr=5e-4,
):
    print(locals())

    # dataset = load_text_files(data_dir)  # Use this if loading raw text

    dataset = datasets.load_from_disk(data_dir)

    if "train" in dataset:
        dataset = dataset["train"]

    # Load model and tokenizer from Hugging Face
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        # revision=revision,
        # attn_implementation="flash_attention_2" if is_bfloat16_supported() else "sdpa",
        torch_dtype=torch.bfloat16 if is_bfloat16_supported() else torch.float32,
        trust_remote_code=True,
    )
    model.cuda()

    if any(keyword in data_dir for keyword in ["code", "cogs", "arithmetic"]):
        collator = DataCollatorWithFlattening()
    else:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        tokenizer.pad_token = tokenizer.eos_token

        collator = DataCollatorForLanguageModeling(
            tokenizer=tokenizer,
            mlm=False,
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
        learning_rate=lr,
        lr_scheduler_type="cosine_with_min_lr",
        lr_scheduler_kwargs={"min_lr_rate": 0.1},
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        data_collator=collator,
    )

    trainer.train()


if __name__ == "__main__":
    fire.Fire(main)
