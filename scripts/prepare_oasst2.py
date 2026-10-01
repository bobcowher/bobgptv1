#!/usr/bin/env python3
"""Build the oasst2 source: human-written English conversations from OpenAssistant.

    python scripts/prepare_oasst2.py

OASST2 (https://huggingface.co/datasets/OpenAssistant/oasst2, Apache-2.0) is a
set of conversation trees written and ranked by volunteers. From each English
tree this keeps one conversation: start at the prompt and repeatedly follow the
best-ranked reply, so the assistant turns are the ones people rated highest.

Dropped: deleted / failed-review / synthetic messages, toxic messages, and
conversations that name Open Assistant or LAION (bobgpt shouldn't learn someone
else's identity). Long conversations are cut back to the longest prefix that
ends on an assistant turn and fits MAX_TOKENS.

Output is data/sources/oasst2/oasst2_en.jsonl, chat records ready for
scripts/build_mix.py.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

import pyarrow.parquet as pq
import tiktoken
from huggingface_hub import hf_hub_download

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from chat_template import render

REPO = "OpenAssistant/oasst2"
REVISION = "179dd21fc55192153d94adb0e0ce8f69e222bf75"
FILES = ("data/train-00000-of-00001-88ba0162028a73fc.parquet",
         "data/validation-00000-of-00001-1deeef95c3248fe0.parquet")
MAX_TOKENS = 1000          # context is 1024
MAX_TOXICITY = 0.5         # detoxify score, per message
OTHER_IDENTITY = re.compile(r"open[\s-]?assistant|laion", re.IGNORECASE)
ROLES = {"prompter": "user", "assistant": "assistant"}


def usable(m: dict) -> bool:
    tox = (m["detoxify"] or {}).get("toxicity") or 0.0
    return (m["lang"] == "en" and not m["deleted"] and m["review_result"] is not False
            and not m["synthetic"] and tox < MAX_TOXICITY)


def best(children: list[dict]) -> dict:
    # rank 0 is the volunteers' top pick; unranked replies sort after ranked ones.
    return min(children, key=lambda m: (m["rank"] is None, m["rank"] or 0, m["created_date"]))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--output-dir", type=Path,
                        default=Path(__file__).resolve().parent.parent / "data" / "sources" / "oasst2")
    args = parser.parse_args()
    enc = tiktoken.get_encoding("gpt2")

    columns = ["message_id", "parent_id", "text", "role", "lang", "review_result", "deleted",
               "rank", "synthetic", "detoxify", "message_tree_id", "created_date"]
    messages = []
    for name in FILES:
        path = hf_hub_download(REPO, name, repo_type="dataset", revision=REVISION)
        messages += pq.read_table(path, columns=columns).to_pylist()

    children = defaultdict(list)
    roots = []
    for m in messages:
        if not usable(m):
            continue
        if m["parent_id"] is None:
            roots.append(m)
        else:
            children[m["parent_id"]].append(m)

    stats = defaultdict(int)
    records = []
    for root in sorted(roots, key=lambda m: m["message_tree_id"]):
        path, node = [root], root
        while children.get(node["message_id"]):
            node = best(children[node["message_id"]])
            path.append(node)
        turns = [{"role": ROLES[m["role"]], "content": m["text"].strip()} for m in path]
        if any(OTHER_IDENTITY.search(t["content"]) for t in turns):
            stats["other identity"] += 1
            continue
        # Longest prefix ending on an assistant turn that fits the context.
        while turns and (turns[-1]["role"] != "assistant"
                         or len(enc.encode_ordinary(render(turns))) > MAX_TOKENS):
            turns.pop()
        if not turns:
            stats["no reply / too long"] += 1
            continue
        stats["kept"] += 1
        stats["truncated"] += len(turns) < len(path)
        records.append({
            "id": f"oasst2/{root['message_tree_id']}",
            "messages": turns,
            "metadata": {"turns": len(turns) // 2, "license": "Apache-2.0",
                         "origin": "OpenAssistant/oasst2 (human-written)"},
        })

    args.output_dir.mkdir(parents=True, exist_ok=True)
    with (args.output_dir / "oasst2_en.jsonl").open("w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    tokens = sum(len(enc.encode_ordinary(render(r["messages"]))) for r in records)
    (args.output_dir / "provenance.json").write_text(json.dumps(
        {"description": "OASST2 English conversations, best-ranked path per tree.",
         "repository": f"https://huggingface.co/datasets/{REPO}", "revision": REVISION,
         "license": "Apache-2.0", "document_count": len(records), "tokens": tokens,
         "stats": dict(stats)}, indent=2) + "\n", encoding="utf-8")
    print(dict(stats), f"{tokens:,} tokens", f"{len(roots)} English roots")


if __name__ == "__main__":
    main()
