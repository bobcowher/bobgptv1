"""Blind A/B rating by hand, to check the judge against your taste.

    python scripts/eval_rate.py [N]      # default 20 pairs

Shows N random pairs gemma has already judged, without run names and in a
random order. Press a, b or t (tie); q stops early. Answers go to
evals/human.jsonl, and eval_rank.py reports how often gemma agreed with you.
"""

import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from eval_common import (HUMAN, PROMPT_VERSION, append_jsonl, load_prompts, load_run,
                         pair_key, read_jsonl)
from eval_judge import render_conversation
from eval_rank import current_verdicts


def main():
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 20
    prompts = {p["id"]: p for p in load_prompts()}
    rated = {pair_key(h["prompt_id"], h["sample"], h["run_a"], h["run_b"]) for h in read_jsonl(HUMAN)}
    pool = [v for v in current_verdicts()
            if pair_key(v["prompt_id"], v["sample"], v["run_a"], v["run_b"]) not in rated]
    runs = {}

    for i, v in enumerate(random.sample(pool, min(n, len(pool))), 1):
        for r in (v["run_a"], v["run_b"]):
            runs.setdefault(r, load_run(r))
        shown = [v["run_a"], v["run_b"]]
        random.shuffle(shown)
        prompt = prompts[v["prompt_id"]]

        print(f"\n{'=' * 70}\n[{i}/{n}]  {prompt['category']}\n")
        print(render_conversation(prompt["messages"]))
        for label, run in zip("AB", shown):
            print(f"\n--- Reply {label} ---\n{runs[run][(v['prompt_id'], v['sample'])]['reply']}")

        key = ""
        while key not in ("a", "b", "t", "q"):
            key = input("\nBetter reply? [a/b/t, q to stop] ").strip().lower()
        if key == "q":
            break
        winner = {"a": shown[0], "b": shown[1]}.get(key)
        append_jsonl(HUMAN, {
            "prompt_version": PROMPT_VERSION, "prompt_id": v["prompt_id"], "sample": v["sample"],
            "run_a": v["run_a"], "run_b": v["run_b"],
            "pick": "tie" if winner is None else "a" if winner == v["run_a"] else "b",
        })


if __name__ == "__main__":
    main()
