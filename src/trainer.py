import torch
from tqdm import tqdm
from torch.utils.data import IterableDataset
import wandb
from transformers import AutoModelForCasualLM

from .utils import (
    get_tokenized_train_dataset,
    get_steps,
    get_train_dataloader,
    create_optimizer_scheduler,
)


class StaticTrainer:
    def __init__(self, args, tokenizer, evaluator):
        self.args = args
        self.tokenizer = tokenizer
        self.evaluator = evaluator
        self.ckpt_steps, self.total_steps = get_steps(self.args)

        self.model = AutoModelForCasualLM.from_pretrained(self.args.model)
        if torch.cuda.is_available():
            self.model = self.model.cuda()

        self.optimizer, self.lr_scheduler = create_optimizer_scheduler(
            self.model,
            args.lr,
            self.total_steps,
            args.inner_opt,
            args.lr_scheduler,
            args.warmup_steps,
        )

    def train(
        self,
        train_data,
        validation_data,
        filter_samples,
        output_idxs,
    ):
        """Standard Pytorch training and evaluation code without any online sampling."""
        n_data = (
            self.args.n_select
            if self.args.n_select != 0
            else self.args.max_steps * self.args.bsz
        )
        tokenized_train = get_tokenized_train_dataset(
            self.args, train_data, n_data, filter_samples
        )

        train_dataloader = get_train_dataloader(
            self.tokenizer, tokenized_train, self.args.bsz
        )

        progress_bar = tqdm(range(self.total_steps))
        counter = 0
        max_grad_norm = 1.0
        self.model.zero_grad()

        num_epochs = (
            1 if isinstance(tokenized_train, IterableDataset) else self.args.n_epochs
        )

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
                loss_all = torch.mean(loss)

                loss_all.backward()

                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_grad_norm)
                self.optimizer.step()

                self.lr_scheduler.step()

                if counter % self.args.K == 0:
                    wandb.log({"train_loss": loss_all}, step=counter)
                    print(f"train_loss: {loss_all}")

                if counter % self.ckpt_steps == 0:
                    self.evaluator.evaluate(
                        self.model,
                        validation_data,
                        counter,
                        None,
                        output_idxs,
                        save=True,
                    )

                self.model.zero_grad()

                counter += 1
                progress_bar.update(1)

                if counter == self.total_steps:
                    break

        self.evaluator.evaluate(
            self.model, validation_data, counter, None, output_idxs, save=True
        )
