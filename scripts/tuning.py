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

# Architecture tuning runs (docs/ARCHITECTURE.md): a 124M model trained on a fixed
# token budget of the pretrain mix, so variants compare at equal tokens.
#   TUNING_NAME    names the run; the checkpoint goes to data/checkpoints/tuning/<name>/
#   TUNING_CONFIG  JSON merged over the 124M config, e.g. '{"norm": "rms"}'
#   SEED           model init and data order; variants share one, the noise-floor
#                  repeat of the baseline changes it
#   TUNING_TOKENS  the budget (default 300M)
#   OPTIMIZER      muon (default: Muon on the blocks' matrices, AdamW on the rest) or adamw
#   GRAD_ACCUM     micro-batches of 16 x 1024 per optimizer step (default 1); warmup and
#                  eval spacing are scaled so they fall at the same token counts
#   MUON_LR        Muon's peak learning rate (default 0.01; 0.005-0.02 were within noise)
name = os.environ.get("TUNING_NAME", "baseline")
overrides = json.loads(os.environ.get("TUNING_CONFIG", "{}"))
seed = int(os.environ.get("SEED", "1"))
budget_tokens = int(float(os.environ.get("TUNING_TOKENS", "300e6")))
grad_accum = int(os.environ.get("GRAD_ACCUM", "1"))
optimizer = os.environ.get("OPTIMIZER", "muon")
muon_lr = float(os.environ.get("MUON_LR", "0.01"))

# No dropout, as in pretrain.py: one pass over fresh data.
cfg = {**GPT_CONFIG_124M, "drop_rate": 0.0, **overrides}
batch_size = 16
max_steps = budget_tokens // (batch_size * cfg["context_length"] * grad_accum)

torch.manual_seed(seed)
train_loader, val_loader = make_loaders("pretrain_v11", cfg, batch_size=batch_size, seed=seed)

model = LanguageModel(gpt_config=cfg,
                      train_loader=train_loader,
                      val_loader=val_loader,
                      checkpoint_path=f"data/checkpoints/tuning/{name}/model.pth",
                      lr=6e-4,  # GPT-2 small's peak
                      compile=True,
                      optimizer=optimizer,
                      muon_lr=muon_lr)
print(f"Tuning run {name}: seed {seed}, {max_steps} steps x {batch_size * cfg['context_length'] * grad_accum} "
      f"tokens = {budget_tokens:,}; {optimizer}{f' lr {muon_lr}' if optimizer == 'muon' else ''}; config {cfg}")
print(f"Parameters: {sum(p.numel() for p in model.model.parameters()):,}")

# The epoch-end full val loss (all of v11's val set) is the number variants are
# compared on; the mid-run evals are a noisy 20-batch sample for the curve.
model.train(num_epochs=1,
            eval_freq=1000 // grad_accum,
            eval_iter=20,
            start_context="Every effort moves you",
            warmup_steps=700 // grad_accum,
            max_steps=max_steps,
            run_name=name,
            grad_accum=grad_accum)
print(f"Peak GPU memory: {torch.cuda.max_memory_allocated() / 2**30:.1f} GiB")
