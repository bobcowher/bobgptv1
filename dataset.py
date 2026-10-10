import json
from pathlib import Path

import numpy as np
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
        if isinstance(chunk, np.ndarray):
            # Built mixes are uint16 memmaps; the model wants int64 ids.
            chunk = torch.from_numpy(chunk.astype(np.int64))
        return chunk[:-1], chunk[1:]

class MaskedGPTDataset(Dataset):
    """GPTDatasetV1 for post-training builds: targets the mask zeroes out become -100,
    which cross_entropy ignores, so only assistant tokens (and replay text) train.
    Windows with no trainable target are dropped -- their loss would be NaN."""

    def __init__(self, token_ids, mask, max_length):
        self.token_ids = token_ids
        self.mask = mask
        self.max_length = max_length
        n = (len(token_ids) - 1) // max_length
        trainable = np.asarray(mask[1:n * max_length + 1]).reshape(n, max_length).any(axis=1)
        self.starts = np.flatnonzero(trainable) * max_length

    def __len__(self):
        return len(self.starts)

    def __getitem__(self, idx):
        start = int(self.starts[idx])
        chunk = torch.from_numpy(self.token_ids[start:start + self.max_length + 1].astype(np.int64))
        keep = torch.from_numpy(self.mask[start + 1:start + self.max_length + 1].astype(bool))
        target = chunk[1:].masked_fill(~keep, -100)
        return chunk[:-1], target


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
                       drop_last=True, num_workers=0, generator=None):
    dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            drop_last=drop_last,
            num_workers=num_workers,
            generator=generator
            )

    return dataloader


def make_loaders(mix_name, cfg, batch_size=2, num_workers=0, seed=None):
    """Loaders over a mix built by scripts/build_mix.py (data/build/<mix_name>/).

    seed fixes the shuffle order with the loaders' own generators, so it doesn't
    depend on how many random numbers model init used: tuning variants with
    different layers still see the same batches in the same order.
    """
    def generator(offset):
        return None if seed is None else torch.Generator().manual_seed(seed + offset)

    root = Path(__file__).resolve().parent
    build_dir = root / "data" / "build" / mix_name
    manifest_path = build_dir / "manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(
            f"{manifest_path} not found. Run: python scripts/build_mix.py {mix_name}")

    # Refuse to train on a build that no longer matches its mix config.
    mix = json.loads((root / "mixes" / f"{mix_name}.json").read_text(encoding="utf-8"))
    if json.loads(manifest_path.read_text(encoding="utf-8"))["mix"] != mix:
        raise RuntimeError(
            f"mixes/{mix_name}.json changed since the last build. "
            f"Run: python scripts/build_mix.py {mix_name}")

    train_ids = np.memmap(build_dir / "train.bin", dtype=np.uint16, mode="r")
    val_ids = np.memmap(build_dir / "val.bin", dtype=np.uint16, mode="r")

    if (build_dir / "train_mask.bin").exists():  # post-training build (loss_mask)
        train_mask = np.memmap(build_dir / "train_mask.bin", dtype=np.uint8, mode="r")
        val_mask = np.memmap(build_dir / "val_mask.bin", dtype=np.uint8, mode="r")
        train_dataset = MaskedGPTDataset(train_ids, train_mask, cfg["context_length"])
        val_dataset = MaskedGPTDataset(val_ids, val_mask, cfg["context_length"])
    else:
        train_dataset = GPTDatasetV1(
                train_ids,
                max_length=cfg["context_length"],
                stride=cfg["context_length"]
                )
        val_dataset = GPTDatasetV1(
                val_ids,
                max_length=cfg["context_length"],
                stride=cfg["context_length"]
                )

    train_loader = _create_dataloader(
            train_dataset,
            batch_size=batch_size,
            drop_last=True,
            shuffle=True,
            num_workers=num_workers,
            generator=generator(0)
            )

    val_loader = _create_dataloader(
            val_dataset,
            batch_size=batch_size,
            drop_last=False,
            shuffle=True,  # mid-epoch evals only read a few batches; make them a random mix
            num_workers=num_workers,
            generator=generator(1)
            )

    return train_loader, val_loader
