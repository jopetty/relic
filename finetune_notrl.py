from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling,
)
import torch
import datasets
import fire

from src.data.utils import get_dataset


def main(
    model_name="EleutherAI/pythia-160m",
    max_seq_length=2048,
    gradient_accumulation_steps=2,
    max_steps=10000,
    bsz=32,
    warmup_steps=1000,
    logging_steps=1,
    save_steps=2500,
    output_dir="outputs",
    seed=3407,
    report_to="wandb",
    lr=5e-4,
):
    dataset = get_slimpj_dataset(seed, is_eval=False, seq_len=max_seq_length)
    # dataset = datasets.load_from_disk(data_dir)

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token

    collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,
    )

    # Load model and tokenizer from Hugging Face
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        attn_implementation="flash_attention_2",
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
    )
    model.cuda()

    # Define training arguments
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

    # Initialize the Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        # If you need custom data collation, add:
        data_collator=collator,
    )

    # Start training
    trainer.train()


if __name__ == "__main__":
    fire.Fire(main)
