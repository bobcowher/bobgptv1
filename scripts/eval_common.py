"""Shared pieces of the ranking eval: prompts, automatic checks, verdicts, Bradley-Terry.

The pipeline (see docs/EVAL.md, "5. Ranking"):
    eval_generate.py  checkpoint -> evals/runs/<run>.jsonl
    eval_judge.py     run vs opponents -> evals/verdicts.jsonl
    eval_rank.py      verdicts -> ranking table
    eval_rate.py      blind A/B by hand -> evals/human.jsonl
Everything here is pure (no model, no network), so it's what the tests cover.
"""

import json
import math
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from chat_template import ROLE_HEADERS  # noqa: E402

EVALS = ROOT / "evals"
RUNS = EVALS / "runs"
VERDICTS = EVALS / "verdicts.jsonl"
HUMAN = EVALS / "human.jsonl"

PROMPT_VERSION = "prompts_v1"
SAMPLES = 3            # sampled replies per prompt per run
BASELINE = "run30"     # anchored at 1000 points
ANCHOR_SCORE = 1000


def read_jsonl(path):
    path = Path(path)
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def append_jsonl(path, record):
    with open(path, "a") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def load_prompts(version=PROMPT_VERSION):
    return read_jsonl(EVALS / f"{version}.jsonl")


def load_run(name):
    """{(prompt_id, sample): record} for one run's generated replies."""
    return {(r["prompt_id"], r["sample"]): r for r in read_jsonl(RUNS / f"{name}.jsonl")}


# ---- automatic checks -------------------------------------------------------

def ngrams(words, n=4):
    return [tuple(words[i:i + n]) for i in range(len(words) - n + 1)]


def checks(reply, finish, last_user):
    """Cheap, deterministic diagnoses of one reply. Not part of the score.

    finish is how generation ended: "end" (### End), "question" (the model
    started a new ### Question turn), "eot", or "length" (ran out of tokens).
    """
    words = reply.lower().split()
    grams = ngrams(words)
    user_grams = set(ngrams(last_user.lower().split()))
    echo_share = sum(g in user_grams for g in grams) / len(grams) if grams else 0.0
    return {
        "stopped": finish in ("end", "eot"),
        # Wrote the user's side: started a new user turn, or put a header in the reply.
        "leak": finish == "question" or any(h in reply for h in ROLE_HEADERS.values()),
        # Copied the user's words: half its 4-grams come from their last message.
        "echo": echo_share >= 0.5,
        # Looped: one 4-gram three or more times.
        "repeat": any(c >= 3 for c in Counter(grams).values()),
    }


# ---- verdicts ---------------------------------------------------------------

def reconcile(first, second):
    """Combine the two orderings of one pair into "a", "b" or "tie".

    first is the judge's pick with run a shown as Reply A; second with run b
    shown as Reply A. Each is "A", "B" or "TIE" in the judge's own labels.
    A win only counts when both orders agree, which cancels position bias.
    """
    as_runs = [{"A": "a", "B": "b"}.get(first, "tie"),
               {"A": "b", "B": "a"}.get(second, "tie")]
    return as_runs[0] if as_runs[0] == as_runs[1] else "tie"


def pair_key(prompt_id, sample, run_a, run_b):
    """Same key whichever run is called a."""
    lo, hi = sorted((run_a, run_b))
    return (prompt_id, sample, lo, hi)


# ---- Bradley-Terry ----------------------------------------------------------

def bradley_terry(matches, iterations=500, prior=0.5):
    """Strengths from (run_x, run_y, result) with result "x", "y" or "tie".

    Minorization-maximization (Hunter 2004). A tie is half a win each way.
    prior adds that many virtual ties between every pair that played, so a run
    that won or lost everything still gets a finite strength.
    Returns {run: strength}, geometric mean 1.
    """
    wins = defaultdict(float)
    games = defaultdict(float)
    for x, y, result in matches:
        key = tuple(sorted((x, y)))
        games[key] += 1
        wins[x] += {"x": 1.0, "y": 0.0, "tie": 0.5}[result]
        wins[y] += {"x": 0.0, "y": 1.0, "tie": 0.5}[result]
    for key in list(games):
        games[key] += 2 * prior
        wins[key[0]] += prior
        wins[key[1]] += prior

    runs = sorted({r for key in games for r in key})
    p = {r: 1.0 for r in runs}
    for _ in range(iterations):
        new = {}
        for r in runs:
            denom = sum(n / (p[r] + p[o]) for (a, b), n in games.items() if r in (a, b)
                        for o in [b if a == r else a])
            new[r] = wins[r] / denom
        log_mean = sum(math.log(v) for v in new.values()) / len(new)
        p = {r: v / math.exp(log_mean) for r, v in new.items()}
    return p


def to_points(strengths, anchor=BASELINE):
    """Elo-like points: 400 * log10 of the strength ratio, anchor at 1000.

    100 points means about a 64% expected win rate.
    """
    base = strengths.get(anchor, 1.0)
    return {r: ANCHOR_SCORE + 400 * math.log10(s / base) for r, s in strengths.items()}


def bootstrap_ci(matches_by_prompt, n=1000, seed=0, anchor=BASELINE):
    """95% intervals on points, resampling prompts (the real unit), not single verdicts."""
    rng = random.Random(seed)
    prompts = list(matches_by_prompt)
    draws = {}
    for _ in range(n):
        sample = [m for pid in rng.choices(prompts, k=len(prompts)) for m in matches_by_prompt[pid]]
        for r, pts in to_points(bradley_terry(sample, iterations=200), anchor).items():
            draws.setdefault(r, []).append(pts)
    return {r: (v[int(0.025 * len(v))], v[int(0.975 * len(v)) - 1])
            for r, v in ((r, sorted(v)) for r, v in draws.items())}
