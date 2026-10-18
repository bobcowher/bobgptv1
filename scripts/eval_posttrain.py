#!/usr/bin/env python3
"""Generate, judge against run 35, and rank a post-trained checkpoint.

    EVAL_RUN=my-run python scripts/eval_posttrain.py

Reads checkpoints/posttrain/model.pth (a pipeline step inherits checkpoints/ from
posttrain). The tracked evals/ is copied to eval/evals/ and the scripts write there
(EVALS_DIR), so the new replies, verdicts and RANKING.md stay with this run; commit
them back into evals/ by hand. EVAL_RUN names the run in the ranking, and in a
pipeline defaults to <pipeline>-<pipeline run number>.
"""
import os
import shutil
import subprocess
import sys
from pathlib import Path

CHECKPOINT = "checkpoints/posttrain/model.pth"
EVALS_DIR = Path("eval/evals")
OPPONENT = "run35"


def main():
    if not os.path.exists(CHECKPOINT):
        sys.exit(f"No checkpoint at {CHECKPOINT}; inherit checkpoints from the posttrain step.")
    run = os.environ.get("EVAL_RUN")
    if not run and "BEEKEEPER_PIPELINE" in os.environ:
        run = f"{os.environ['BEEKEEPER_PIPELINE']}-{os.environ['BEEKEEPER_PIPELINE_RUN_ID']}"
    if not run:
        sys.exit("Set EVAL_RUN to name this run in the ranking.")

    shutil.copytree("evals", EVALS_DIR, dirs_exist_ok=True)
    env = {**os.environ, "EVALS_DIR": str(EVALS_DIR)}
    for args in (["eval_generate.py", run, CHECKPOINT],
                 ["eval_judge.py", run, OPPONENT],
                 ["eval_rank.py"]):
        subprocess.run([sys.executable, f"scripts/{args[0]}", *args[1:]], env=env, check=True)
    print(f"Wrote {EVALS_DIR}")


if __name__ == "__main__":
    main()
