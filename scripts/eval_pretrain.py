#!/usr/bin/env python3
"""Score a pretrained checkpoint on the mix val splits and the frozen sets.

    python scripts/eval_pretrain.py

Reads checkpoints/pretrain/model.pth (a pipeline step inherits checkpoints/ from
pretrain) and writes everything it prints to eval/pretrain.txt. EVAL_MAX_TOKENS
caps eval_mix_val's tokens per source; a smoke test sets it small.
"""
import os
import subprocess
import sys
from pathlib import Path

CHECKPOINT = "checkpoints/pretrain/model.pth"
OUT = Path("eval/pretrain.txt")

# Each command is one eval script plus the environment it needs on top of ours.
EVALS = [
    ("eval_mix_val.py", {}),
    ("eval_mix_val.py", {"EVAL_MIX": "pretrain_decay_v3"}),
    ("eval_frozen.py", {}),
    ("eval_frozen.py", {"EVAL_CTX": "1024"}),
]


def main():
    # Fail loudly: scoring some other checkpoint would look like a result.
    if not os.path.exists(CHECKPOINT):
        sys.exit(f"No checkpoint at {CHECKPOINT}; inherit checkpoints from the pretrain step.")
    OUT.parent.mkdir(exist_ok=True)
    with OUT.open("w") as out:
        for script, env in EVALS:
            result = subprocess.run([sys.executable, f"scripts/{script}", CHECKPOINT],
                                    env={**os.environ, **env}, capture_output=True, text=True)
            print(result.stdout, end="")
            out.write(result.stdout)
            if result.returncode != 0:
                sys.exit(f"{script} {env} failed:\n{result.stderr}")
    print(f"Wrote {OUT}")


if __name__ == "__main__":
    main()
