from unsloth import FastLanguageModel, is_bfloat16_supported

from trl import SFTTrainer
from transformers import TrainingArguments, Trainer
import torch
import datasets

from src.data.utils import load_text_files
import fire


def main(
    # data_dir="./data/crasp/depth9",
    data_dir="./data/tokenized/depth9_llama_10k",
    model_name="unsloth/Llama-3.2-1B",
    max_seq_length=2048,
    gradient_accumulation_steps=1,
    max_steps=5000,
    bsz=16,
    warmup_steps=500,
    logging_steps=1,
    save_steps=250,
    output_dir="outputs",
    optim="adamw_8bit",
    seed=3407,
    report_to="wandb",
    use_rslora=False,
    loftq_config=None,
):
    # dataset = load_text_files(data_dir)
    dataset = datasets.load_from_disk(data_dir)

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_name,
        max_seq_length=max_seq_length,
        dtype=None,
    )

    # cache tokenizer state? opted to tokenization caching independently, but this is
    # allegedly another option
    _ = tokenizer("Dummy text", truncation=True)

    model = FastLanguageModel.get_peft_model(
        model,
        r=16,
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
        lora_alpha=32,
        lora_dropout=0,
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=seed,
        max_seq_length=max_seq_length,
        use_rslora=use_rslora,
        loftq_config=loftq_config,
    )

    trainer = SFTTrainer(
        model=model,
        train_dataset=dataset,
        dataset_text_field="text",
        max_seq_length=max_seq_length,
        # tokenizer=tokenizer,
        args=TrainingArguments(
            per_device_train_batch_size=bsz,
            gradient_accumulation_steps=gradient_accumulation_steps,
            warmup_steps=warmup_steps,
            max_steps=max_steps,
            fp16=not is_bfloat16_supported(),
            bf16=is_bfloat16_supported(),
            logging_steps=logging_steps,
            save_strategy="steps",
            save_steps=save_steps,
            output_dir=output_dir,
            optim=optim,
            seed=seed,
            report_to=report_to,
        ),
    )

    trainer.train()


if __name__ == "__main__":
    fire.Fire(main)
