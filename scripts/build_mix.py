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

from chat_template import render

SOURCES_DIR = ROOT / "data" / "sources"
BUILD_DIR = ROOT / "data" / "build"
MIXES_DIR = ROOT / "mixes"
HASH_BUCKETS = 1000


def is_val(unit_id: str, val_fraction: float) -> bool:
    bucket = int(hashlib.sha256(unit_id.encode()).hexdigest()[:8], 16) % HASH_BUCKETS
    return bucket < val_fraction * HASH_BUCKETS


def record_text(record: dict) -> str:
    if "messages" in record:
        return render(record["messages"])
    return record["text"]


def encode(tokenizer, texts: list[str], eot: int, batch: int = 10_000) -> list[np.ndarray]:
    """Token ids per document, EOT-terminated, as uint16 arrays (Python int lists won't fit in RAM)."""
    docs = []
    for start in range(0, len(texts), batch):
        for ids in tokenizer.encode_ordinary_batch(texts[start:start + batch]):
            ids.append(eot)
            docs.append(np.array(ids, dtype=np.uint16))
    return docs


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


def load_units(source: str, chunk_chars: int):
    """Yield (unit_id, text) for every document (or document part) in a source."""
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

    splits = {"train": [], "val": []}
    summary = {}
    for source in mix["sources"]:
        name, repeat = source["name"], source.get("repeat", 1)
        # A big source can take a smaller val share so the full-val pass stays fast.
        val_fraction = source.get("val_fraction", mix["val_fraction"])
        units = list(load_units(name, mix["chunk_chars"]))
        val_texts = [text for unit_id, text in units if is_val(unit_id, val_fraction)]
        train_texts = [text for unit_id, text in units if not is_val(unit_id, val_fraction)]
        del units

        # Val is never repeated: it measures the data, not the mix weights.
        val_tokens = encode(tokenizer, val_texts, eot)
        train_tokens = encode(tokenizer, train_texts, eot)
        splits["val"].extend(val_tokens)
        splits["train"].extend(train_tokens * repeat)

        summary[name] = {
            "repeat": repeat,
            "train_units": len(train_texts),
            "val_units": len(val_texts),
            "train_tokens": sum(map(len, train_tokens)) * repeat,
            "val_tokens": sum(map(len, val_tokens)),
        }
        s = summary[name]
        print(f"{name:<12} units train {s['train_units']:>6,} val {s['val_units']:>5,}   "
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
