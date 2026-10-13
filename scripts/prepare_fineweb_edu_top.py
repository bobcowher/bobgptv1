#!/usr/bin/env python3
"""Keep the highest-scoring FineWeb-Edu pages from all of sample-10BT as a JSONL source.

    python scripts/prepare_fineweb_edu_top.py --tokens 2_400_000_000

prepare_fineweb_edu.py takes a random slice of one shard. This reads every
sample-10BT shard (14 x ~2.1GB, cached by huggingface_hub), finds the
classifier `score` above which the pages add up to --tokens, and keeps those.
FineWeb-Edu already dropped pages below int_score 3, so this filters further
within the educational set rather than separating good pages from spam.

Records use the same "fineweb_edu/<id>" ids as prepare_fineweb_edu.py, so
build_mix's hashed val split puts any page that is in pretrain_v11's val in
this source's val too, as long as the mix gives it the same val_fraction.

Output is data/sources/fineweb_edu_top/sample10bt_top_<shard>.jsonl.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
from huggingface_hub import hf_hub_download

from prepare_fineweb_edu import COLUMNS, REPO

SHARDS = 14


def shard_path(shard: int) -> str:
    return hf_hub_download(REPO, f"sample/10BT/{shard:03d}_00000.parquet", repo_type="dataset")


def threshold(paths: list[str], tokens: int) -> float:
    """Lowest score to keep so the kept pages total about `tokens` tokens."""
    scores, counts = [], []
    for path in paths:
        table = pq.read_table(path, columns=["score", "token_count"])
        scores.append(table.column("score").to_numpy())
        counts.append(table.column("token_count").to_numpy())
    scores, counts = np.concatenate(scores), np.concatenate(counts)
    order = np.argsort(-scores, kind="stable")
    cumulative = np.cumsum(counts[order])
    cut = min(np.searchsorted(cumulative, tokens), len(order) - 1)
    print(f"{len(scores):,} pages, {cumulative[-1]:,} tokens; "
          f"top {cumulative[cut]:,} tokens are score >= {scores[order[cut]]:.4f}")
    return float(scores[order[cut]])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--tokens", type=int, default=2_400_000_000,
                        help="approximate GPT-2 tokens to keep")
    parser.add_argument("--output-dir", type=Path,
                        default=Path(__file__).resolve().parent.parent / "data" / "sources" / "fineweb_edu_top")
    args = parser.parse_args()

    paths = [shard_path(shard) for shard in range(SHARDS)]
    min_score = threshold(paths, args.tokens)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    docs = tokens = 0
    for shard, path in enumerate(paths):
        out_path = args.output_dir / f"sample10bt_top_{shard:03d}.jsonl"
        with out_path.open("w", encoding="utf-8") as f:
            for batch in pq.ParquetFile(path).iter_batches(columns=COLUMNS, batch_size=10_000):
                for row in batch.to_pylist():
                    if row["score"] < min_score:
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
        print(f"shard {shard}: running total {docs:,} documents, ~{tokens:,} tokens")

    print(f"Wrote {docs:,} documents, ~{tokens:,} tokens to {args.output_dir}")


if __name__ == "__main__":
    main()
