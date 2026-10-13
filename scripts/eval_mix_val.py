#!/usr/bin/env python3
"""Score checkpoints on a built mix's val split, per source and per group.

    python scripts/eval_mix_val.py data/checkpoints/tuning/a/model.pth data/checkpoints/tuning/b/model.pth

Data-mix tuning runs can't be compared on their own val loss: each mix has its own
val split. This scores every checkpoint on one fixed split (pretrain_v11 unless
EVAL_MIX says otherwise), so a mix that helps code but hurts web text shows as both.
Each source's val documents are recovered from val.bin by replaying build_mix.py's
shuffle (see source_slices).

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
import tiktoken
import torch

from config import config_from_state_dict
from languagemodel import LanguageModel

MIX = os.environ.get("EVAL_MIX", "pretrain_v11")
# A cap keeps FineWeb (25M val tokens) from taking most of the time; 2M tokens
# puts the standard error well under the differences between runs.
MAX_TOKENS = int(float(os.environ.get("EVAL_MAX_TOKENS", "2e6")))
EOT = tiktoken.get_encoding("gpt2").eot_token
GROUPS = {
    "web": ("fineweb_edu",),
    "code": ("github_python", "github_cpp", "github_rust", "github_shell"),
    "docs": ("python_docs", "python_books", "code_docs", "linux_manpages"),
    "books": ("books",),
}


def source_slices(build_dir):
    """{source: uint16 token array} of each source's val documents.

    build_mix.py shuffles documents across sources before writing val.bin, so this
    replays its shuffle: default_rng(0) permutes the train documents, then the val
    documents, and every document ends in one <|endoftext|>. Exact per-source token
    counts against the manifest prove the replay matched.
    """
    manifest = json.loads((build_dir / "manifest.json").read_text(encoding="utf-8"))
    sources = manifest["sources"]
    rng = np.random.default_rng(0)
    rng.permutation(sum(s["train_units"] * s["repeat"] for s in sources.values()))
    order = rng.permutation(sum(s["val_units"] for s in sources.values()))
    # owner[i]: the source of document i in build order (sources in mix order).
    owner = np.repeat(np.arange(len(sources)), [s["val_units"] for s in sources.values()])

    val = np.array(np.memmap(build_dir / "val.bin", dtype=np.uint16, mode="r"))
    ends = np.flatnonzero(val == EOT) + 1
    assert len(ends) == len(order) and ends[-1] == len(val), "val.bin doesn't match the manifest"
    docs = np.split(val, ends[:-1])
    slices = {}
    for k, name in enumerate(sources):
        # Written position p holds build-order document order[p].
        slices[name] = np.concatenate([docs[p] for p in np.flatnonzero(owner[order] == k)])
        assert len(slices[name]) == sources[name]["val_tokens"], f"{name}: shuffle replay failed"
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
