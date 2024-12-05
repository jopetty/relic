import torch
from tqdm import tqdm
import wandb
from transformers import AutoModelForCausalLM, DataCollatorForLanguageModeling

from .utils import create_optimizer_scheduler


def compute_kl_loss(model, reference_model, input_ids, attention_mask=None):
    """
    Compute KL divergence loss between model and reference model outputs
    Args:
        model: current model
        reference_model: frozen reference model
        input_ids: input token ids
        attention_mask: optional attention mask
    Returns:
        KL divergence loss

    Notes:
    The KL divergence is unbounded, but all sequences here are the same length
    because we are pretraining. If you are using sequences of different lengths,
    you should consider normalizing the KL divergence by the sequence length.
    """
    with torch.no_grad():
        ref_logits = reference_model(input_ids, attention_mask=attention_mask).logits
        ref_probs = torch.nn.functional.log_softmax(ref_logits, dim=-1)

    logits = model(input_ids, attention_mask=attention_mask).logits
    log_probs = torch.nn.functional.log_softmax(logits, dim=-1)

    kl_loss = torch.nn.functional.kl_div(
        log_probs, ref_probs, reduction="batchmean", log_target=True
    )
    return kl_loss


class StaticTrainer:
    def __init__(self, args, tokenizer):
        self.args = args
        self.tokenizer = tokenizer
        self.ckpt_steps, self.total_steps = self.args.eval_every, self.args.total_steps

        self.model = AutoModelForCausalLM.from_pretrained(
            args.model_name,
            # attn_implementation="flash_attention_2",
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
            device_map=0,
        )

        # copy model as reference model
        self.reference_model = AutoModelForCausalLM.from_pretrained(
            args.model_name,
            # attn_implementation="flash_attention_2",
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
            device_map=0,
        )

        self.optimizer, self.lr_scheduler = create_optimizer_scheduler(
            self.model,
            args.lr,
            self.total_steps,
            args.lr_scheduler,
            args.warmup_steps,
        )

    def train(
        self,
        train_data,
        validation_data,
    ):
        """Standard Pytorch training and evaluation code without any online sampling."""
        data_collator = DataCollatorForLanguageModeling(self.tokenizer, mlm=False)
        train_dataloader = torch.utils.data.DataLoader(
            train_data,
            batch_size=self.args.bsz,
            collate_fn=data_collator,
            drop_last=False,
            num_workers=0,
            pin_memory=True,
        )

        progress_bar = tqdm(range(self.total_steps))
        counter = 0
        max_grad_norm = 1.0
        self.model.zero_grad()

        num_epochs = 1  # iterable dataset

        for epoch in range(num_epochs):
            for i, batch in enumerate(train_dataloader):
                self.model.train()
                input_ids = batch["input_ids"]
                labels = batch["labels"]

                if torch.cuda.is_available():
                    input_ids = input_ids.cuda()
                    labels = labels.cuda()

                outputs = self.model(input_ids=input_ids, labels=labels)
                loss = outputs.loss

                if self.args.kl_loss:
                    kl_loss = compute_kl_loss(
                        self.model, self.reference_model, input_ids
                    )
                    print(f"Step {counter}, Loss: {loss}, KL Loss: {kl_loss}")
                    loss += self.args.kl_weight * kl_loss

                loss.backward()

                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_grad_norm)
                self.optimizer.step()

                self.lr_scheduler.step()

                if counter % self.args.eval_every == 0:
                    wandb.log({"train_loss": loss}, step=counter)

                self.model.zero_grad()

                counter += 1
                progress_bar.update(1)

                if counter == self.total_steps:
                    break
