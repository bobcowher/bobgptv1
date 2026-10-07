"""Rank runs from the judge's verdicts, for the ranking eval.

    python scripts/eval_rank.py

Fits Bradley-Terry to every verdict on the current prompt set and judge
prompt, anchored at run30 = 1000 (100 points is about a 64% win rate), with
95% intervals from resampling prompts. Overlapping intervals mean "tied".
Prints the table and writes it to evals/RANKING.md, along with the automatic
checks and, once you've rated pairs with eval_rate.py, how often gemma agreed
with you.
"""

import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from eval_common import (BASELINE, EVALS, HUMAN, PROMPT_VERSION, RUNS, VERDICTS, bootstrap_ci,
                         bradley_terry, load_prompts, pair_key, read_jsonl, to_points)
from eval_judge import JUDGE_MODEL, JUDGE_VERSION, OPUS_MODEL

CHECKS = ("stopped", "leak", "echo", "repeat")


def current_verdicts(judge=JUDGE_MODEL):
    return [v for v in read_jsonl(VERDICTS)
            if v["prompt_version"] == PROMPT_VERSION and v["judge_version"] == JUDGE_VERSION
            and v["judge"] == judge]


def winners(verdicts):
    """{pair_key: winning run, or None for a tie}."""
    return {pair_key(v["prompt_id"], v["sample"], v["run_a"], v["run_b"]):
            {"a": v["run_a"], "b": v["run_b"]}.get(v["result"]) for v in verdicts}


def human_winners():
    return {pair_key(h["prompt_id"], h["sample"], h["run_a"], h["run_b"]):
            {"a": h["run_a"], "b": h["run_b"]}.get(h["pick"])
            for h in read_jsonl(HUMAN) if h["prompt_version"] == PROMPT_VERSION}


def check_rates(run):
    records = read_jsonl(RUNS / f"{run}.jsonl")
    return {c: sum(r["checks"][c] for r in records) / len(records) for c in CHECKS}


def category_win_rates(verdicts, categories):
    """{run: {category: share of points won}} with a tie worth half."""
    points, games = defaultdict(float), defaultdict(int)
    for v in verdicts:
        cat = categories[v["prompt_id"]]
        for run, side in ((v["run_a"], "a"), (v["run_b"], "b")):
            games[run, cat] += 1
            points[run, cat] += 1.0 if v["result"] == side else 0.5 if v["result"] == "tie" else 0.0
    rates = defaultdict(dict)
    for (run, cat), n in games.items():
        rates[run][cat] = points[run, cat] / n
    return rates


def agreement(reference, judge):
    """Compare a judge with a reference rater on the pairs both judged.

    Returns (agreed, compared, tied): agreement where both picked a winner,
    and how often the judge tied a pair the reference had a winner for
    (signal the judge threw away).
    """
    agreed = compared = tied = 0
    for key, ref in reference.items():
        if ref is None or key not in judge:
            continue
        if judge[key] is None:
            tied += 1
            continue
        compared += 1
        agreed += judge[key] == ref
    return agreed, compared, tied


def agreement_line(reference_name, reference, judge_name, judge):
    agreed, compared, tied = agreement(reference, judge)
    if not compared + tied:
        return f"- {judge_name} vs {reference_name}: no shared pairs yet."
    return (f"- {judge_name} vs {reference_name}: of {compared + tied} pairs where {reference_name} "
            f"picked a winner, {judge_name} tied {tied}; on the other {compared} it agreed {agreed}"
            + (f" ({agreed / compared:.0%})." if compared else "."))


def main():
    verdicts = current_verdicts()
    if not verdicts:
        sys.exit("No verdicts yet; run scripts/eval_judge.py first.")
    categories = {p["id"]: p["category"] for p in load_prompts()}
    cats = list(dict.fromkeys(categories.values()))

    by_prompt = defaultdict(list)
    for v in verdicts:
        by_prompt[v["prompt_id"]].append(
            (v["run_a"], v["run_b"], {"a": "x", "b": "y", "tie": "tie"}[v["result"]]))
    matches = [m for ms in by_prompt.values() for m in ms]
    anchor = BASELINE if any(BASELINE in m[:2] for m in matches) else matches[0][0]
    points = to_points(bradley_terry(matches), anchor)
    ci = bootstrap_ci(by_prompt, anchor=anchor)
    cat_rates = category_win_rates(verdicts, categories)

    games = defaultdict(int)
    split = defaultdict(int)
    for v in verdicts:
        for run in (v["run_a"], v["run_b"]):
            games[run] += 1
            # The two orders disagreed: the judge's pick followed position, not content.
            split[run] += v["result"] == "tie" and not (v["first"] == v["second"] == "TIE")

    header = (["run", "score", "95% CI", "matches", "split"] + list(CHECKS) + cats)
    rows = []
    for run in sorted(points, key=points.get, reverse=True):
        rates = check_rates(run)
        rows.append([run, f"{points[run]:.0f}", f"{ci[run][0]:.0f}–{ci[run][1]:.0f}",
                     str(games[run]), f"{split[run] / games[run]:.0%}"]
                    + [f"{rates[c]:.0%}" for c in CHECKS]
                    + [f"{cat_rates[run][c]:.0%}" if c in cat_rates[run] else "" for c in cats])

    gemma, opus, robert = winners(verdicts), winners(current_verdicts(OPUS_MODEL)), human_winners()
    lines = [
        f"# bobgpt ranking ({PROMPT_VERSION}, judge {JUDGE_MODEL} v{JUDGE_VERSION})",
        "",
        "Generated by `scripts/eval_rank.py`; see docs/EVAL.md, \"5. Ranking\".",
        f"Score: Bradley-Terry, {anchor} = 1000, 100 points ≈ 64% win rate. "
        "Overlapping 95% intervals mean tied. Category columns are share of points won.",
        "",
        "| " + " | ".join(header) + " |",
        "|" + "---|" * len(header),
        *("| " + " | ".join(r) + " |" for r in rows),
        "",
        "Judge agreement (Opus and Robert judge samples, blind; see docs/EVAL.md):",
        agreement_line("Robert", robert, "Opus", opus),
        agreement_line("Opus", opus, "gemma", gemma),
        agreement_line("Robert", robert, "gemma", gemma),
    ]
    (EVALS / "RANKING.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
