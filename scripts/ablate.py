import json
import os
import sys
from pathlib import Path

import torch

# Make the project root importable when run as scripts/<name>.py
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import GPT_CONFIG_124M
from dataset import make_loaders
from languagemodel import LanguageModel

# Architecture ablations (docs/ARCHITECTURE.md): a 124M model trained on a fixed
# token budget of the pretrain mix, so variants compare at equal tokens.
#   ABLATE_NAME    names the run; the checkpoint goes to data/checkpoints/ablate/<name>/
#   ABLATE_CONFIG  JSON merged over the 124M config, e.g. '{"norm": "rms"}'
#   SEED           model init and data order; variants share one, the noise-floor
#                  repeat of the baseline changes it
#   ABLATE_TOKENS  the budget (default 300M)
name = os.environ.get("ABLATE_NAME", "baseline")
overrides = json.loads(os.environ.get("ABLATE_CONFIG", "{}"))
seed = int(os.environ.get("SEED", "1"))
budget_tokens = int(float(os.environ.get("ABLATE_TOKENS", "300e6")))

# No dropout, as in pretrain.py: one pass over fresh data.
cfg = {**GPT_CONFIG_124M, "drop_rate": 0.0, **overrides}
batch_size = 16
max_steps = budget_tokens // (batch_size * cfg["context_length"])

torch.manual_seed(seed)
train_loader, val_loader = make_loaders("pretrain_v11", cfg, batch_size=batch_size, seed=seed)

model = LanguageModel(gpt_config=cfg,
                      train_loader=train_loader,
                      val_loader=val_loader,
                      checkpoint_path=f"data/checkpoints/ablate/{name}/model.pth",
                      lr=6e-4,  # GPT-2 small's peak
                      compile=True)
print(f"Ablation {name}: seed {seed}, {max_steps} steps x {batch_size * cfg['context_length']} "
      f"tokens = {budget_tokens:,}; config {cfg}")
print(f"Parameters: {sum(p.numel() for p in model.model.parameters()):,}")

# The epoch-end full val loss (all of v11's val set) is the number variants are
# compared on; the mid-run evals are a noisy 20-batch sample for the curve.
model.train(num_epochs=1,
            eval_freq=1000,
            eval_iter=20,
            start_context="Every effort moves you",
            warmup_steps=700,
            max_steps=max_steps)
print(f"Peak GPU memory: {torch.cuda.max_memory_allocated() / 2**30:.1f} GiB")
