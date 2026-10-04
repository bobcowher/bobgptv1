#!/usr/bin/env python3
"""Build a training mix: source JSONL -> train/val token files.

    python scripts/build_mix.py pretrain_v1

Reads mixes/<name>.json, loads every data/sources/<source>/*.jsonl it names,
splits each document into train or val, tokenizes, and writes
data/build/<name>/{train,val}.bin (uint16 GPT-2 token ids, documents separated
by <|endoftext|>) plus manifest.json describing what went in.

Split rule: a unit goes to val when a hash of its id lands in the bottom
`val_fraction` of buckets. Adding new documents never moves existing ones
between train and val. Documents longer than `chunk_chars` are cut into parts
at paragraph breaks first, and each part is hashed on its own -- otherwise the
25 books would split as 2-3 whole novels in val.

Post-training mixes set "loss_mask": true. The build then also writes
{train,val}_mask.bin (uint8, one byte per token): 1 where the model should learn
the token (assistant replies, see chat_template.render_with_spans, and all of any
plain-text source), 0 for system/user turns. Chat records are not chunked.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import tiktoken

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from chat_template import render, render_with_spans

SOURCES_DIR = ROOT / "data" / "sources"
BUILD_DIR = ROOT / "data" / "build"
MIXES_DIR = ROOT / "mixes"
HASH_BUCKETS = 1000


def is_val(unit_id: str, val_fraction: float) -> bool:
    bucket = int(hashlib.sha256(unit_id.encode()).hexdigest()[:8], 16) % HASH_BUCKETS
    return bucket < val_fraction * HASH_BUCKETS


def is_kept(unit_id: str, fraction: float) -> bool:
    """Hash-based subsample, salted so it is independent of the val split."""
    bucket = int(hashlib.sha256(f"keep:{unit_id}".encode()).hexdigest()[:8], 16) % 1_000_000
    return bucket < fraction * 1_000_000


def record_text(record: dict) -> str:
    if "messages" in record:
        return render(record["messages"])
    return record["text"]


def record_text_spans(record: dict):
    """(text, trainable char spans); spans is None when every token is trainable."""
    if "messages" in record:
        return render_with_spans(record["messages"])
    return record["text"], None


def encode(tokenizer, texts: list[str], eot: int, batch: int = 10_000) -> list[np.ndarray]:
    """Token ids per document, EOT-terminated, as uint16 arrays (Python int lists won't fit in RAM)."""
    docs = []
    for start in range(0, len(texts), batch):
        for ids in tokenizer.encode_ordinary_batch(texts[start:start + batch]):
            ids.append(eot)
            docs.append(np.array(ids, dtype=np.uint16))
    return docs


def encode_masks(tokenizer, units: list[tuple[str, list | None]], docs: list[np.ndarray]) -> list[np.ndarray]:
    """Per-token loss masks matching encode()'s output: a token is trainable when it starts in a span."""
    masks = []
    for (text, spans), ids in zip(units, docs):
        if spans is None:
            masks.append(np.ones(len(ids), dtype=np.uint8))
            continue
        _, offsets = tokenizer.decode_with_offsets(ids[:-1].tolist())
        starts = np.array(offsets)
        mask = np.zeros(len(ids), dtype=np.uint8)
        for a, b in spans:
            mask[:-1] |= (starts >= a) & (starts < b)
        mask[-1] = 1  # the EOT after END_MARKER
        masks.append(mask)
    return masks


def chunk(text: str, chunk_chars: int) -> list[str]:
    parts = []
    while len(text) > chunk_chars:
        cut = text.find("\n\n", chunk_chars)
        if cut == -1:
            break
        parts.append(text[:cut])
        text = text[cut:].lstrip("\n")
    parts.append(text)
    return parts


def load_units(source: str, chunk_chars: int, with_spans: bool = False):
    """Yield (unit_id, text) for every document (or document part) in a source.

    with_spans yields (unit_id, (text, spans)) instead, and never chunks chat records.
    """
    files = sorted((SOURCES_DIR / source).glob("*.jsonl"))
    if not files:
        raise SystemExit(f"No JSONL files in {SOURCES_DIR / source}")
    ids = set()
    for path in files:
        with path.open(encoding="utf-8") as f:
            for line in f:
                record = json.loads(line)
                if record["id"] in ids:
                    raise SystemExit(f"Duplicate id {record['id']!r} in source {source}")
                ids.add(record["id"])
                if with_spans:
                    text, spans = record_text_spans(record)
                    if spans is not None:
                        yield record["id"], (text, spans)
                        continue
                    parts = [(p, None) for p in chunk(text, chunk_chars)]
                    if len(parts) == 1:
                        yield record["id"], parts[0]
                    else:
                        for i, part in enumerate(parts):
                            yield f"{record['id']}#part{i:03d}", part
                    continue
                parts = chunk(record_text(record), chunk_chars)
                if len(parts) == 1:
                    yield record["id"], parts[0]
                else:
                    for i, part in enumerate(parts):
                        yield f"{record['id']}#part{i:03d}", part


def build(mix_name: str) -> None:
    mix_path = MIXES_DIR / f"{mix_name}.json"
    mix = json.loads(mix_path.read_text(encoding="utf-8"))
    tokenizer = tiktoken.get_encoding("gpt2")
    eot = tokenizer.eot_token
    loss_mask = mix.get("loss_mask", False)

    splits = {"train": [], "val": []}
    masks = {"train": [], "val": []}
    summary = {}
    for source in mix["sources"]:
        name, repeat = source["name"], source.get("repeat", 1)
        # A big source can take a smaller val share so the full-val pass stays fast.
        val_fraction = source.get("val_fraction", mix["val_fraction"])
        # fraction < 1 keeps a fixed hash-selected share of a big source's units.
        fraction = source.get("fraction", 1.0)
        units = [u for u in load_units(name, mix["chunk_chars"], loss_mask) if is_kept(u[0], fraction)]
        val_units = [unit for unit_id, unit in units if is_val(unit_id, val_fraction)]
        train_units = [unit for unit_id, unit in units if not is_val(unit_id, val_fraction)]
        del units
        val_texts = [u[0] for u in val_units] if loss_mask else val_units
        train_texts = [u[0] for u in train_units] if loss_mask else train_units

        # Val is never repeated: it measures the data, not the mix weights.
        val_tokens = encode(tokenizer, val_texts, eot)
        train_tokens = encode(tokenizer, train_texts, eot)
        splits["val"].extend(val_tokens)
        splits["train"].extend(train_tokens * repeat)
        if loss_mask:
            masks["val"].extend(encode_masks(tokenizer, val_units, val_tokens))
            masks["train"].extend(encode_masks(tokenizer, train_units, train_tokens) * repeat)

        summary[name] = {
            "repeat": repeat,
            "fraction": fraction,
            "train_units": len(train_texts),
            "val_units": len(val_texts),
            "train_tokens": sum(map(len, train_tokens)) * repeat,
            "val_tokens": sum(map(len, val_tokens)),
        }
        s = summary[name]
        print(f"{name:<14} units train {s['train_units']:>6,} val {s['val_units']:>5,}   "
              f"tokens train {s['train_tokens']:>11,} val {s['val_tokens']:>10,}")

    out_dir = BUILD_DIR / mix_name
    out_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(0)
    for split, docs in splits.items():
        # Shuffle document order so no source sits in one long contiguous run.
        order = rng.permutation(len(docs))
        with (out_dir / f"{split}.bin").open("wb") as f:
            for i in order:
                docs[i].tofile(f)
        if loss_mask:
            with (out_dir / f"{split}_mask.bin").open("wb") as f:
                for i in order:
                    masks[split][i].tofile(f)

    manifest = {
        "mix": mix,
        "tokenizer": "gpt2",
        "dtype": "uint16",
        "sources": summary,
        "train_tokens": sum(s["train_tokens"] for s in summary.values()),
        "val_tokens": sum(s["val_tokens"] for s in summary.values()),
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"\nWrote {manifest['train_tokens']:,} train / {manifest['val_tokens']:,} val tokens "
          f"to {out_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("mix", help="name of a file in mixes/, without .json")
    build(parser.parse_args().mix)


if __name__ == "__main__":
    main()
