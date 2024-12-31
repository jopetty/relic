import numpy as np
import torch
import os
import datasets

from .utils import get_slimpj_dataset


class InfiniteShuffledDataset(torch.utils.data.IterableDataset):
    def __init__(self, file_dir, max_steps):
        self.data = datasets.load_from_disk(file_dir).remove_columns(["text"])
        self.data_order = None
        self.current_idx = 0
        self.length = max_steps

    def __iter__(self):
        while True:  # Infinite iteration
            if self.data_order is None or self.current_idx >= len(self.data):
                self.data_order = np.random.permutation(len(self.data))
                self.current_idx = 0

            item = self.data[int(self.data_order[self.current_idx])]
            self.current_idx += 1
            yield item

    def __len__(self):
        return self.length


class HybridDataset(torch.utils.data.IterableDataset):
    def __init__(self, seed, is_eval, sampling_prob, seq_len, synthetic_dir):
        self.seed = seed
        self.is_eval = is_eval
        self.seq_len = seq_len

        self.slimpj_dataset = get_slimpj_dataset(seed, is_eval, seq_len)
        self.synthetic_dataset = InfiniteShuffledDataset(synthetic_dir)
        self.sampling_prob = sampling_prob

    def __iter__(self):
        # return item from slimpj_dataset with probability sampling_prob
        slimpj_iter = iter(self.slimpj_dataset)
        synthetic_iter = iter(self.synthetic_dataset)
        while True:
            if np.random.rand() > self.sampling_prob:
                input_ids = next(slimpj_iter)["input_ids"].astype(np.int64)
                input_ids = torch.tensor(input_ids)
                attn_mask = torch.ones_like(input_ids, dtype=torch.int64)

                yield {"input_ids": input_ids, "attention_mask": attn_mask}
            else:
                yield next(synthetic_iter)


def main(seed, is_eval, seq_len, synthetic_dir):
    dataset = HybridDataset(seed, is_eval, 0.5, seq_len, synthetic_dir)
    for i, item in enumerate(dataset):
        print(item)
        if i == 10:
            break


if __name__ == "__main__":
    import fire

    fire.Fire(main)
