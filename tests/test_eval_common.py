import random
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

from config import GPT_CONFIG_124M, GPT_CONFIG_406M, config_from_state_dict
from eval_common import bradley_terry, checks, pair_key, reconcile, to_points
from models import GPTModel


# ---- checks -----------------------------------------------------------------

def test_clean_reply_passes_every_check():
    c = checks("Hi! What can I help you with today?", "end", "Hey Bobgpt.")
    assert c == {"stopped": True, "leak": False, "echo": False, "repeat": False}


def test_running_out_of_tokens_is_not_stopped():
    assert not checks("A list is", "length", "q")["stopped"]


def test_starting_the_users_turn_is_a_leak():
    assert checks("Sure.", "question", "q")["leak"]
    assert checks("Sure.\n\n### Answer\nMore.", "end", "q")["leak"]


def test_markdown_headings_are_not_a_leak():
    assert not checks("### Step 1\nInstall it.", "end", "q")["leak"]


def test_copying_the_user_is_echo():
    user = "can you explain how python decorators work please"
    assert checks("Can you explain how Python decorators work please?", "end", user)["echo"]


def test_a_loop_is_repeat():
    reply = "A tuple is a tuple of values. " * 3
    assert checks(reply, "end", "q")["repeat"]


# ---- verdicts ---------------------------------------------------------------

def test_reconcile_needs_both_orders_to_agree():
    # first: run a shown as Reply A. second: run b shown as Reply A.
    assert reconcile("A", "B") == "a"     # a won both times
    assert reconcile("B", "A") == "b"     # b won both times
    assert reconcile("A", "A") == "tie"   # the judge just picked the first reply
    assert reconcile("A", "TIE") == "tie"
    assert reconcile("garbled", "B") == "tie"


def test_pair_key_ignores_order():
    assert pair_key("p1", 0, "run32", "run30") == pair_key("p1", 0, "run30", "run32")


# ---- Bradley-Terry ----------------------------------------------------------

def simulate(strengths, games_per_pair, seed=0):
    rng = random.Random(seed)
    runs = list(strengths)
    matches = []
    for i, x in enumerate(runs):
        for y in runs[i + 1:]:
            p_x = strengths[x] / (strengths[x] + strengths[y])
            for _ in range(games_per_pair):
                matches.append((x, y, "x" if rng.random() < p_x else "y"))
    return matches


def test_bradley_terry_recovers_known_order_and_gaps():
    # Points 1000, 1100, 1300 on the scale to_points uses.
    true = {"run30": 1.0, "mid": 10 ** (100 / 400), "best": 10 ** (300 / 400)}
    points = to_points(bradley_terry(simulate(true, 2000)))
    assert points["run30"] == 1000
    assert points["best"] > points["mid"] > points["run30"]
    assert abs(points["mid"] - 1100) < 25
    assert abs(points["best"] - 1300) < 25


def test_hundred_points_is_about_64_percent():
    matches = [("a", "b", "x")] * 64 + [("a", "b", "y")] * 36
    points = to_points(bradley_terry(matches, prior=0), anchor="b")
    assert abs(points["a"] - 1100) < 5


def test_ties_pull_runs_together():
    points = to_points(bradley_terry([("a", "b", "tie")] * 50), anchor="b")
    assert abs(points["a"] - 1000) < 1e-6


def test_clean_sweep_stays_finite():
    points = to_points(bradley_terry([("a", "b", "x")] * 10), anchor="b")
    assert 1000 < points["a"] < 2000


# ---- model size from a checkpoint -------------------------------------------

def test_config_from_state_dict_matches_each_config():
    for cfg in (GPT_CONFIG_124M, GPT_CONFIG_406M):
        with torch.device("meta"):
            state = GPTModel(cfg).state_dict()
        assert config_from_state_dict(state) == {**cfg, "drop_rate": 0.0}
