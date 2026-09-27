#!/usr/bin/env python3
"""Take a slice of FineWeb-Edu (sample-10BT) as a JSONL source.

    python scripts/prepare_fineweb_edu.py --tokens 200_000_000

FineWeb-Edu is Common Crawl web text filtered for educational value
(https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu, ODC-By 1.0).
sample-10BT is split into ~750M-token parquet shards. This downloads one shard
(~2.1GB, cached by huggingface_hub) and keeps a hash-selected fraction of its
documents, so the slice is spread across the whole shard and rerunning with the
same arguments gives the same documents.

Output is data/sources/fineweb_edu/sample10bt_<shard>.jsonl, one record per
page, ready for scripts/build_mix.py.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import pyarrow.parquet as pq
from huggingface_hub import hf_hub_download

REPO = "HuggingFaceFW/fineweb-edu"
COLUMNS = ["id", "text", "url", "dump", "score", "token_count"]


def keep(doc_id: str, fraction: float) -> bool:
    bucket = int(hashlib.sha256(doc_id.encode()).hexdigest()[:8], 16) % 1_000_000
    return bucket < fraction * 1_000_000


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--tokens", type=int, default=200_000_000,
                        help="approximate GPT-2 tokens to keep")
    parser.add_argument("--shard", type=int, default=0, help="which sample-10BT shard to read")
    parser.add_argument("--output-dir", type=Path,
                        default=Path(__file__).resolve().parent.parent / "data" / "sources" / "fineweb_edu")
    args = parser.parse_args()

    path = hf_hub_download(REPO, f"sample/10BT/{args.shard:03d}_00000.parquet", repo_type="dataset")
    shard = pq.ParquetFile(path)

    # token_count is FineWeb's own GPT-2 count, so the fraction can be set up front.
    shard_tokens = sum(
        batch.column("token_count").to_numpy().sum()
        for batch in shard.iter_batches(columns=["token_count"])
    )
    fraction = min(1.0, args.tokens / shard_tokens)
    print(f"shard {args.shard}: {shard_tokens:,} tokens, keeping ~{fraction:.1%}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    out_path = args.output_dir / f"sample10bt_{args.shard:03d}.jsonl"
    docs = tokens = 0
    with out_path.open("w", encoding="utf-8") as f:
        for batch in shard.iter_batches(columns=COLUMNS, batch_size=10_000):
            for row in batch.to_pylist():
                if not keep(row["id"], fraction):
                    continue
                record = {
                    "id": f"fineweb_edu/{row['id']}",
                    "text": row["text"],
                    "metadata": {
                        "url": row["url"],
                        "dump": row["dump"],
                        "edu_score": row["score"],
                        "license": "ODC-By-1.0",
                    },
                }
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
                docs += 1
                tokens += row["token_count"]

    print(f"Wrote {docs:,} documents, ~{tokens:,} tokens to {out_path}")


if __name__ == "__main__":
    main()
