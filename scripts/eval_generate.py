"""Generate a run's replies to the frozen prompt set, for the ranking eval.

    python scripts/eval_generate.py run32 checkpoints/run32_model.pth

Writes evals/runs/<run>.jsonl: SAMPLES seeded replies per prompt, produced the
way api.py serves them (same prompt rendering, top_k, stop handling), plus the
automatic checks. Generate once per checkpoint; judging reuses the file.
"""

import sys
import zlib
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from chat_template import END_MARKER, render_prompt
from config import config_from_state_dict
from eval_common import PROMPT_VERSION, RUNS, SAMPLES, append_jsonl, checks, load_prompts
from languagemodel import LanguageModel
from text_stream import STOP_STRINGS, TextStream

# What Open WebUI gets from api.py, except temperature: api.py defaults to 1.0,
# 0.7 is a more typical chat setting and what we tune for.
TEMPERATURE = 0.7
TOP_K = 40
MAX_TOKENS = 300


def generate(model, messages, seed):
    """One reply, streamed through TextStream exactly as the API does."""
    torch.manual_seed(seed)
    encoded = model.text_to_token_ids(render_prompt(messages), model.tokenizer).to(model.device)
    stream = TextStream(model.tokenizer)
    reply, n = "", 0
    with torch.no_grad():
        for token_id in model.generate_streaming(idx=encoded,
                                                 max_new_tokens=MAX_TOKENS,
                                                 context_size=model.model.pos_emb.weight.shape[0],
                                                 temperature=TEMPERATURE,
                                                 top_k=TOP_K,
                                                 eos_id=model.tokenizer.eot_token):
            n += 1
            reply += stream.push(token_id.item())
            if stream.stopped:
                break
    reply += stream.finish()

    if stream.stopped:
        # Whichever stop string came first: "### End" is a clean finish, a new
        # "### Question" means the model started writing the user's turn.
        cuts = {s: stream.text.find(s) for s in STOP_STRINGS if s in stream.text}
        finish = "end" if min(cuts, key=cuts.get) == END_MARKER else "question"
    else:
        finish = "length" if n == MAX_TOKENS else "eot"
    return reply.strip(), finish, n


def main():
    run, checkpoint = sys.argv[1], sys.argv[2]
    out = RUNS / f"{run}.jsonl"
    if out.exists():
        sys.exit(f"{out} exists; delete it to regenerate.")

    model = LanguageModel(gpt_config=config_from_state_dict(torch.load(checkpoint, map_location="cpu")))
    model.load_the_model(checkpoint)
    model.model.eval()

    RUNS.mkdir(parents=True, exist_ok=True)
    for prompt in load_prompts():
        for sample in range(SAMPLES):
            # Same seed for every run, so sample k of two runs gets the same draws.
            seed = zlib.crc32(f"{prompt['id']}:{sample}".encode())
            reply, finish, tokens = generate(model, prompt["messages"], seed)
            append_jsonl(out, {
                "prompt_version": PROMPT_VERSION, "prompt_id": prompt["id"], "sample": sample,
                "run": run, "checkpoint": checkpoint, "seed": seed,
                "temperature": TEMPERATURE, "top_k": TOP_K, "max_tokens": MAX_TOKENS,
                "reply": reply, "finish": finish, "tokens": tokens,
                "checks": checks(reply, finish, prompt["messages"][-1]["content"]),
            })
        print(f"{prompt['id']:12s} {reply[:90]!r}")


if __name__ == "__main__":
    main()
