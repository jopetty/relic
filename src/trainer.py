import torch
import torch.nn.functional as F
from tqdm import tqdm
import wandb
from copy import deepcopy
import os

from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    DataCollatorForLanguageModeling,
)

from .utils import create_optimizer_scheduler


def kl_div(logits, ref_logits):
    ref_probs = torch.nn.functional.log_softmax(ref_logits, dim=-1)
    log_probs = torch.nn.functional.log_softmax(logits, dim=-1)

    return torch.nn.functional.kl_div(
        log_probs, ref_probs, reduction="batchmean", log_target=True
    )
    # normalize by num logits
    # retur


def kd_loss(
    logits: torch.Tensor, teacher_logits: torch.Tensor, label: torch.Tensor
) -> torch.Tensor:
    teacher_probs = F.softmax(teacher_logits, dim=-1, dtype=torch.float32)
    inf_mask = torch.isinf(logits)
    logprobs = F.log_softmax(logits, dim=-1, dtype=torch.float32)
    prod_probs = torch.masked_fill(teacher_probs * logprobs, inf_mask, 0)
    x = torch.sum(prod_probs, dim=-1).view(-1)
    mask = (label != -100).int()
    distil_loss = -torch.sum(x * mask.view(-1), dim=0) / torch.sum(mask.view(-1), dim=0)

    return distil_loss


class StaticTrainer:
    def __init__(self, args, tokenizer, output_dir):
        self.args = args
        self.tokenizer = tokenizer
        self.output_dir = output_dir

        if args.reinit:
            assert (
                not args.shrink_and_perturb
            ), "Cannot reinit from scratch and shrink and perturb"
            config = AutoConfig.from_pretrained(args.model_name)
            self.model = AutoModelForCausalLM.from_config(
                config,
                attn_implementation="flash_attention_2",
                torch_dtype=torch.bfloat16,
                trust_remote_code=True,
            )
            self.model.cuda()
        else:
            self.model = AutoModelForCausalLM.from_pretrained(
                args.model_name,
                attn_implementation="flash_attention_2",
                torch_dtype=torch.bfloat16,
                trust_remote_code=True,
                device_map=0,
            )

        if args.shrink_and_perturb:
            config = AutoConfig.from_pretrained(args.model_name)
            random_init = AutoModelForCausalLM.from_config(
                config,
                attn_implementation="flash_attention_2",
                torch_dtype=torch.bfloat16,
                trust_remote_code=True,
            )
            random_init.cuda()

            # average current model weights with random_init weights
            for param, random_param in zip(
                self.model.parameters(), random_init.parameters()
            ):
                param.data = (
                    param.data * args.lamb + random_param.data + args.noise_scale
                )

        # copy model as reference model
        if args.kl_loss:
            self.reference_model = deepcopy(self.model)
            self.reference_model.eval()

        self.optimizer, self.lr_scheduler = create_optimizer_scheduler(
            self.model,
            args.lr,
            args.total_steps,
            args.optimizer,
            args.lr_scheduler,
            args.warmup_steps,
        )

        # save args to save dir
        with open(os.path.join(output_dir, wandb.run.name, "args.txt"), "w") as f:
            f.write(str(args))

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

        progress_bar = tqdm(range(self.args.total_steps))
        counter = 0
        max_grad_norm = 1.0
        self.model.zero_grad()
        self.model.train()

        for i, batch in enumerate(train_dataloader):
            input_ids = batch["input_ids"].cuda()
            labels = batch["labels"].cuda()

            outputs = self.model(input_ids=input_ids, labels=labels)
            lm_loss = outputs.loss

            if self.args.kl_loss:
                with torch.no_grad():
                    teacher_logits = self.reference_model(input_ids).logits
                kl_loss = kd_loss(outputs.logits, teacher_logits, labels)

                loss = self.args.kl_weight * kl_loss + lm_loss
            else:
                loss = lm_loss

            loss.backward()

            if (i + 1) % self.args.gradient_accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_grad_norm)
                self.optimizer.step()
                self.lr_scheduler.step()
                self.model.zero_grad()

                counter += 1
                progress_bar.update(1)

            if counter % self.args.eval_every == 0:
                if self.args.kl_loss:
                    wandb.log(
                        {
                            "train_loss": lm_loss,
                            "kl_loss": kl_loss,
                        },
                        step=counter,
                    )
                else:
                    wandb.log({"train_loss": loss}, step=counter)

            # save every
            if counter % self.args.save_every == 0:
                self.save_model(counter)

            if counter == self.args.total_steps:
                break

        self.save_model(counter)

    def save_model(self, counter):
        save_path = os.path.join(
            self.output_dir, wandb.run.name, f"checkpoint-{counter}"
        )
        os.makedirs(save_path, exist_ok=True)
        self.model.save_pretrained(save_path)
        torch.save(self.optimizer.state_dict(), os.path.join(save_path, "optimizer.pt"))
        torch.save(
            self.lr_scheduler.state_dict(),
            os.path.join(save_path, "scheduler.pt"),
        )
