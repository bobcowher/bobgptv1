#!/usr/bin/env python3
"""Score checkpoints on a built mix's val split, per source and per group.

    python scripts/eval_mix_val.py data/checkpoints/tuning/a/model.pth data/checkpoints/tuning/b/model.pth

Data-mix tuning runs can't be compared on their own val loss: each mix has its own
val split. This scores every checkpoint on one fixed split (pretrain_v11 unless
EVAL_MIX says otherwise), so a mix that helps code but hurts web text shows as both.
build_mix.py writes val.bin source by source in the mix's order, and the manifest
records each source's val token count, so a source's tokens are one slice.

The split is by document id hash (build_mix.is_val), so a later mix that keeps each
source's val_fraction never trains on these documents.
"""

import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np
import torch

from config import config_from_state_dict
from languagemodel import LanguageModel

MIX = os.environ.get("EVAL_MIX", "pretrain_v11")
# A cap keeps FineWeb (25M val tokens) from taking most of the time; 2M tokens
# puts the standard error well under the differences between runs.
MAX_TOKENS = int(float(os.environ.get("EVAL_MAX_TOKENS", "2e6")))
GROUPS = {
    "web": ("fineweb_edu",),
    "code": ("github_python", "github_cpp", "github_rust", "github_shell"),
    "docs": ("python_docs", "python_books", "code_docs", "linux_manpages"),
    "books": ("books",),
}


def source_slices(build_dir):
    """{source: uint16 token array} in the order build_mix.py wrote them."""
    manifest = json.loads((build_dir / "manifest.json").read_text(encoding="utf-8"))
    val = np.memmap(build_dir / "val.bin", dtype=np.uint16, mode="r")
    slices, start = {}, 0
    for source, stats in manifest["sources"].items():
        end = start + stats["val_tokens"]
        slices[source] = val[start:end]
        start = end
    assert start == len(val), f"manifest covers {start:,} of {len(val):,} val tokens"
    return slices


@torch.no_grad()
def score(model, tokens, ctx, batch_size=16):
    """Mean per-token loss over non-overlapping ctx-token windows, and the token count."""
    n = min(len(tokens) - 1, MAX_TOKENS) // ctx
    ids = torch.from_numpy(tokens[: n * ctx + 1].astype(np.int64))
    x, y = ids[:-1].view(n, ctx), ids[1:].view(n, ctx)
    total = 0.0
    for i in range(0, n, batch_size):
        total += model.calc_loss_batch(x[i:i + batch_size], y[i:i + batch_size]).item() * len(x[i:i + batch_size])
    return total / n, n * ctx


def main():
    slices = source_slices(ROOT / "data" / "build" / MIX)
    for checkpoint in sys.argv[1:]:
        # Build each model at the size it was trained with.
        cfg = config_from_state_dict(torch.load(checkpoint, map_location="cpu"))
        model = LanguageModel(gpt_config=cfg)
        model.load_the_model(checkpoint)
        model.model.eval()
        losses = {s: score(model, t, cfg["context_length"]) for s, t in slices.items()}
        # Groups weight each source by the tokens scored, i.e. a group's loss per token.
        groups = {}
        for group, sources in GROUPS.items():
            scored = [losses[s] for s in sources if s in losses]
            groups[group] = sum(l * n for l, n in scored) / sum(n for _, n in scored)
        print(checkpoint)
        print("  " + "  ".join(f"{g} {l:.3f}" for g, l in groups.items()))
        print("  " + "  ".join(f"{s} {l:.3f}" for s, (l, _) in losses.items()))


if __name__ == "__main__":
    main()
