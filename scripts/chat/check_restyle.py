#!/usr/bin/env python3
"""Check restyled chat batches against their neutral originals.

    python check_restyle.py            # every batch with an original in batches_neutral/

A restyle may only change assistant turns: ids, record count, system and user
turns must match the original exactly. Prints one line per batch and exits 1
if any batch drifted.
"""

import json
import sys
from pathlib import Path

HERE = Path(__file__).absolute().parent


def load(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def fixed_parts(record):
    return record["id"], [(m["role"], m["content"] if m["role"] != "assistant" else None)
                          for m in record["messages"]]


def main() -> None:
    bad = 0
    for original in sorted((HERE / "batches_neutral").glob("*.jsonl")):
        restyled = HERE / "batches" / original.name
        if not restyled.exists():
            print(f"{original.stem}: not restyled yet")
            continue
        old, new = load(original), load(restyled)
        drift = sum(fixed_parts(a) != fixed_parts(b) for a, b in zip(old, new)) + abs(len(old) - len(new))
        changed = sum(a["messages"] != b["messages"] for a, b in zip(old, new))
        print(f"{original.stem}: {len(new)} records, {changed} rewritten, {drift} drifted")
        bad += drift > 0
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
