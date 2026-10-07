"""Judge one run against others with gemma (or Opus), for the ranking eval.

    ssh -N -L 18080:localhost:8080 lab &      # gemma lives behind llama-swap on lab
    python scripts/eval_judge.py run32 run30 run31
    python scripts/eval_judge.py --judge opus --sample 30 run32 run30

gemma judges every pair and is what the ranking uses. Opus (headless
`claude -p`, on the plan) judges a seeded random sample of pairs as the
reference rater in Robert's place; eval_rank.py reports how often gemma
agrees with it.

Compares the first run with each of the others, prompt by prompt, sample k
against sample k. Each pair is judged twice with the replies swapped, and only
a pick both orders agree on counts as a win (eval_common.reconcile).
Appends to evals/verdicts.jsonl and skips pairs already judged, so it can be
stopped and restarted.
"""

import json
import os
import random
import re
import subprocess
import sys
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from eval_common import (HUMAN, PROMPT_VERSION, SAMPLES, VERDICTS, append_jsonl, load_prompts, load_run,
                         pair_key, read_jsonl, reconcile)

JUDGE_URL = os.environ.get("JUDGE_URL", "http://localhost:18080/v1/chat/completions")
JUDGE_MODEL = "gemma-4-12b"
OPUS_MODEL = "claude-opus-5-5"
# Bump when JUDGE_PROMPT or the call settings change: verdicts from different
# judges aren't mixed.
JUDGE_VERSION = 2
# v1 said "TIE when equally good or equally bad" and tied 6 of 9 pairs Robert
# had a clear pick on: at this model size both replies are usually bad.
# llama.cpp serves several requests at once: 4 in flight is ~2x the throughput of 1.
WORKERS = 4

JUDGE_PROMPT = """You are comparing two replies from bobgpt, a small language model trained from scratch as a hobby project. Neither reply will be as good as a large assistant's. Decide which one is better for the user, not whether either is perfect.

A better reply:
- responds to the user's LAST message, in the context of the conversation
- is correct (facts, code, commands)
- is coherent and does not repeat itself
- stays in its own role: it does not write the user's side of the conversation
- has a sensible length for the request

What a good reply does here: {good}

Conversation so far:
{conversation}

Reply A:
{a}

Reply B:
{b}

Both replies will often be poor. Even then, one is usually closer to a good answer: more relevant to the request, partly correct, or more coherent. Pick that one. Use TIE only if you truly cannot separate them.

Think briefly, then end with exactly one line: VERDICT: A, VERDICT: B, or VERDICT: TIE."""


def render_conversation(messages):
    names = {"user": "User", "assistant": "bobgpt", "system": "System"}
    return "\n\n".join(f"{names[m['role']]}: {m['content']}" for m in messages)


def ask_opus(prompt):
    # A minimal system prompt and no tools or settings: with Claude Code's own
    # system prompt a call costs ~20x more.
    result = subprocess.run(
        ["claude", "-p", "--model", OPUS_MODEL, "--tools", "", "--setting-sources", "",
         "--strict-mcp-config", "--system-prompt", "You judge chatbot replies."],
        input=prompt, capture_output=True, text=True, timeout=300)
    return result.stdout


def parse(text):
    picks = re.findall(r"VERDICT:\s*\**\s*(A|B|TIE)\b", text, re.IGNORECASE)
    return (picks[-1].upper() if picks else "NONE"), text


def ask(prompt, judge_model=JUDGE_MODEL):
    if judge_model == OPUS_MODEL:
        return parse(ask_opus(prompt))
    # Hidden thinking off: with it, gemma spends ~700 tokens (~20s) per call and
    # sometimes runs out before the verdict. The prompt asks for brief visible
    # reasoning instead, which costs ~130 tokens.
    body = json.dumps({"model": JUDGE_MODEL, "temperature": 0, "max_tokens": 1024,
                       "chat_template_kwargs": {"enable_thinking": False},
                       "messages": [{"role": "user", "content": prompt}]}).encode()
    req = urllib.request.Request(JUDGE_URL, body, {"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=300) as resp:
        return parse(json.load(resp)["choices"][0]["message"]["content"])


def judge(prompt, reply_a, reply_b, judge_model=JUDGE_MODEL):
    return ask(JUDGE_PROMPT.format(good=prompt["good"],
                                   conversation=render_conversation(prompt["messages"]),
                                   a=reply_a or "(empty reply)", b=reply_b or "(empty reply)"),
               judge_model)


def main():
    args = sys.argv[1:]
    judge_model, n_pairs = JUDGE_MODEL, None
    while args[0].startswith("--"):
        flag, value, args = args[0], args[1], args[2:]
        if flag == "--judge":
            judge_model = {"gemma": JUDGE_MODEL, "opus": OPUS_MODEL}[value]
        elif flag == "--sample":
            n_pairs = int(value)   # judge only this many random pairs per opponent
    run, opponents = args[0], args[1:]
    prompts = load_prompts()
    done = {pair_key(v["prompt_id"], v["sample"], v["run_a"], v["run_b"])
            for v in read_jsonl(VERDICTS)
            if v["prompt_version"] == PROMPT_VERSION and v["judge_version"] == JUDGE_VERSION
            and v["judge"] == judge_model}

    replies = load_run(run)
    for opponent in opponents:
        other = load_run(opponent)
        jobs = [(prompt, k) for prompt in prompts for k in range(SAMPLES)]
        if n_pairs is not None:
            # Seeded, so a rerun draws the same pairs and skips the ones already done.
            # Pairs Robert rated by hand always go in: they calibrate the sampled judge.
            rated = {pair_key(h["prompt_id"], h["sample"], h["run_a"], h["run_b"]) for h in read_jsonl(HUMAN)}
            drawn = random.Random(f"{run}:{opponent}").sample(jobs, min(n_pairs, len(jobs)))
            jobs = drawn + [j for j in jobs if j not in drawn
                            and pair_key(j[0]["id"], j[1], run, opponent) in rated]
        jobs = [(prompt, k) for prompt, k in jobs if pair_key(prompt["id"], k, run, opponent) not in done]

        def both_orders(job):
            prompt, sample = job
            mine, theirs = replies[prompt["id"], sample]["reply"], other[prompt["id"], sample]["reply"]
            return job, judge(prompt, mine, theirs, judge_model), judge(prompt, theirs, mine, judge_model)

        tally = {"a": 0, "b": 0, "tie": 0}
        with ThreadPoolExecutor(WORKERS) as pool:
            for i, ((prompt, sample), (first, first_text), (second, second_text)) in enumerate(
                    pool.map(both_orders, jobs), 1):
                result = reconcile(first, second)
                tally[result] += 1
                append_jsonl(VERDICTS, {
                    "prompt_version": PROMPT_VERSION, "judge": judge_model,
                    "judge_version": JUDGE_VERSION,
                    "prompt_id": prompt["id"], "sample": sample, "run_a": run, "run_b": opponent,
                    "first": first, "second": second, "result": result,
                    "first_reason": first_text, "second_reason": second_text,
                })
                if i % 30 == 0 or i == len(jobs):
                    print(f"{run} vs {opponent}  {i}/{len(jobs)}  "
                          f"{run} {tally['a']}  {opponent} {tally['b']}  tie {tally['tie']}", flush=True)

if __name__ == "__main__":
    main()
