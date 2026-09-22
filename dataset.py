import torch
import tiktoken
from torch.utils.data import Dataset, DataLoader

class GPTDatasetV1(Dataset):

    def __init__(self, token_ids, max_length, stride):
        self.token_ids = token_ids
        self.max_length = max_length
        self.stride = stride
        self.num_samples = max(
            0,
            (len(token_ids) - max_length - 1) // stride + 1,
        )

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        start = idx * self.stride
        chunk = self.token_ids[start:start + self.max_length + 1]
        return chunk[:-1], chunk[1:]

def create_dataloader_v1(txt, batch_size=4, max_length=256,
                         stride=128, shuffle=True, drop_last=True,
                         num_workers=0):
    tokenizer = tiktoken.get_encoding("gpt2")
    token_ids = torch.tensor(tokenizer.encode(txt), dtype=torch.long)
    dataset = GPTDatasetV1(token_ids, max_length, stride)
    return _create_dataloader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            drop_last=drop_last,
            num_workers=num_workers
            )


def _create_dataloader(dataset, batch_size=4, shuffle=True,
                       drop_last=True, num_workers=0):
    dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            drop_last=drop_last,
            num_workers=num_workers
            )

    return dataloader


def make_loaders(text, cfg, train_ratio=0.90, batch_size=2, num_workers=0):
    tokenizer = tiktoken.get_encoding("gpt2")
    token_ids = torch.tensor(tokenizer.encode(text), dtype=torch.long)
    split_idx = int(train_ratio * len(token_ids))

    train_dataset = GPTDatasetV1(
            token_ids[:split_idx],
            max_length=cfg["context_length"],
            stride=cfg["context_length"]
            )
    val_dataset = GPTDatasetV1(
            token_ids[split_idx:],
            max_length=cfg["context_length"],
            stride=cfg["context_length"]
            )

    train_loader = _create_dataloader(
            train_dataset,
            batch_size=batch_size,
            drop_last=True,
            shuffle=True,
            num_workers=num_workers
            )

    val_loader = _create_dataloader(
            val_dataset,
            batch_size=batch_size,
            drop_last=False,
            shuffle=False,
            num_workers=num_workers
            )

    return train_loader, val_loader
