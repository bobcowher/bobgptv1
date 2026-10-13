import json
import os
import sys
from pathlib import Path

import torch

# Make the project root importable when run as scripts/<name>.py
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import GPT_CONFIG_124M
from dataset import SwitchLoader, make_loaders
from languagemodel import LanguageModel, wsd_decay_start

# Architecture tuning runs (docs/ARCHITECTURE.md): a 124M model trained on a fixed
# token budget of the pretrain mix, so variants compare at equal tokens.
#   TUNING_NAME    names the run; the checkpoint goes to data/checkpoints/tuning/<name>/
#   TUNING_CONFIG  JSON merged over the 124M config, e.g. '{"norm": "rms"}'
#   SEED           model init and data order; variants share one, the noise-floor
#                  repeat of the baseline changes it
#   TUNING_TOKENS  the budget (default 300M)
#   LR_SCHEDULE    wsd (default: hold the peak, linear decay over the last 20%) or cosine
#   MIX            the training mix (default pretrain_v11)
#   DECAY_MIX      a second mix for WSD's decay phase (default: none, MIX throughout)
# Val is always pretrain_v11's, so data-mix variants compare on the same documents.
name = os.environ.get("TUNING_NAME", "baseline")
overrides = json.loads(os.environ.get("TUNING_CONFIG", "{}"))
seed = int(os.environ.get("SEED", "1"))
budget_tokens = int(float(os.environ.get("TUNING_TOKENS", "300e6")))
lr_schedule = os.environ.get("LR_SCHEDULE", "wsd")
mix = os.environ.get("MIX", "pretrain_v11")
decay_mix = os.environ.get("DECAY_MIX")
if decay_mix and lr_schedule != "wsd":
    raise ValueError("DECAY_MIX needs LR_SCHEDULE=wsd")

# No dropout, as in pretrain.py: one pass over fresh data.
cfg = {**GPT_CONFIG_124M, "drop_rate": 0.0, **overrides}
batch_size = 16
max_steps = budget_tokens // (batch_size * cfg["context_length"])

torch.manual_seed(seed)
train_loader, val_loader = make_loaders("pretrain_v11", cfg, batch_size=batch_size, seed=seed)
if mix != "pretrain_v11":
    train_loader, _ = make_loaders(mix, cfg, batch_size=batch_size, seed=seed)
if decay_mix:
    decay_loader, _ = make_loaders(decay_mix, cfg, batch_size=batch_size, seed=seed)
    train_loader = SwitchLoader(train_loader, decay_loader, wsd_decay_start(max_steps))

model = LanguageModel(gpt_config=cfg,
                      train_loader=train_loader,
                      val_loader=val_loader,
                      checkpoint_path=f"data/checkpoints/tuning/{name}/model.pth",
                      lr=6e-4,  # GPT-2 small's peak
                      compile=True)
print(f"Tuning run {name}: seed {seed}, {max_steps} steps x {batch_size * cfg['context_length']} "
      f"tokens = {budget_tokens:,}; {lr_schedule}; mix {mix}{f' then {decay_mix}' if decay_mix else ''}; config {cfg}")
print(f"Parameters: {sum(p.numel() for p in model.model.parameters()):,}")

# The epoch-end full val loss (all of v11's val set) is the number variants are
# compared on; the mid-run evals are a noisy 20-batch sample for the curve.
model.train(num_epochs=1,
            eval_freq=1000,
            eval_iter=20,
            start_context="Every effort moves you",
            warmup_steps=700,
            max_steps=max_steps,
            run_name=name,
            lr_schedule=lr_schedule)
print(f"Peak GPU memory: {torch.cuda.max_memory_allocated() / 2**30:.1f} GiB")
