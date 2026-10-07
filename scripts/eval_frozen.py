#!/usr/bin/env python3
"""Score checkpoints on the frozen books / python_docs benchmark.

    python scripts/eval_frozen.py checkpoints/run21_model.pth checkpoints/run22_model.pth

The benchmark is the pretrain_v1 val split of books and python_docs (val_fraction
0.1, chunk_chars 20000). The split depends only on document ids, so no later mix
trains on these units, and the numbers stay comparable across runs whatever the
current training mix is. Reports mean per-token loss per source, always in
256-token windows (see CTX).

Caveat: runs before the sources/mixes restructure (<= run 19) used a different
split and trained on most of these units; their scores are not valid.
"""

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import tiktoken
import torch

from build_mix import is_val, load_units
from config import config_from_state_dict
from languagemodel import LanguageModel

SOURCES = ("books", "python_docs")
# python_docs as it was at pretrain_v1. Projects added later are excluded so the
# benchmark keeps measuring the same documents.
FROZEN_PREFIXES = {
    "python_docs": tuple(f"{p}/" for p in (
        "cpython", "python_peps", "flask", "click", "requests", "rich", "attrs", "fastapi", "black")),
}
VAL_FRACTION, CHUNK_CHARS = 0.1, 20000
# Scored in fixed 256-token windows whatever the model's context length, so runs
# trained at 1024 are compared with earlier 256-context runs on equal terms.
# EVAL_CTX=1024 scores long-context runs at their own length (not comparable).
CTX = int(os.environ.get("EVAL_CTX", 256))


def val_tokens(source, enc):
    tokens = []
    for unit_id, text in load_units(source, CHUNK_CHARS):
        if is_val(unit_id, VAL_FRACTION) and unit_id.startswith(FROZEN_PREFIXES.get(source, "")):
            tokens.extend(enc.encode_ordinary(text) + [enc.eot_token])
    return torch.tensor(tokens)


@torch.no_grad()
def score(model, tokens, batch_size=32):
    n = (len(tokens) - 1) // CTX
    x = tokens[: n * CTX].view(n, CTX)
    y = tokens[1: n * CTX + 1].view(n, CTX)
    total = 0.0
    for i in range(0, n, batch_size):
        logits = model.model(x[i:i + batch_size].to(model.device))
        total += torch.nn.functional.cross_entropy(
            logits.flatten(0, 1), y[i:i + batch_size].to(model.device).flatten(),
            reduction="sum").item()
    return total / (n * CTX)


def main():
    enc = tiktoken.get_encoding("gpt2")
    data = {source: val_tokens(source, enc) for source in SOURCES}
    print("  ".join(f"{s}: {len(t):,} tokens" for s, t in data.items()))
    for checkpoint in sys.argv[1:]:
        # Build each model at the size it was trained with.
        cfg = config_from_state_dict(torch.load(checkpoint, map_location="cpu"))
        model = LanguageModel(gpt_config=cfg)
        model.load_the_model(checkpoint)
        model.model.eval()
        print(f"{checkpoint}  " + "  ".join(f"{s} {score(model, t):.3f}" for s, t in data.items()))


if __name__ == "__main__":
    main()
