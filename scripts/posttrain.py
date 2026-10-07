import os
import sys
from pathlib import Path

import torch

# Make the project root importable when run as scripts/<name>.py
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import GPT_CONFIG_406M
from dataset import make_loaders
from languagemodel import LanguageModel, PRETRAIN_CHECKPOINT, POSTTRAIN_CHECKPOINT


# Post-training starts from pretrained weights: pretrain.py's output by default,
# or any checkpoint named by INIT_CHECKPOINT (e.g. an older run's model).
init_checkpoint = os.environ.get("INIT_CHECKPOINT", PRETRAIN_CHECKPOINT)
# Experiments set these on the Beekeeper project so they don't overwrite the
# checkpoint bobgpt serves (POSTTRAIN_CHECKPOINT) or change the default mix.
mix = os.environ.get("POSTTRAIN_MIX", "posttrain_v2")
output_checkpoint = os.environ.get("POSTTRAIN_OUT", POSTTRAIN_CHECKPOINT)

# Fail loudly if the weights are missing or don't fit: fine-tuning a random
# model would look like a run.
if not os.path.exists(init_checkpoint):
    sys.exit(f"No pretrained checkpoint at {init_checkpoint}; run scripts/pretrain.py first.")

# Must match the pretrained checkpoint's architecture (pretrain.py's config).
# An INIT_CHECKPOINT from a 124M run (<= 30) needs GPT_CONFIG_124M here.
cfg = GPT_CONFIG_406M

# Q&A + chat with loss on assistant replies only (see mixes/<mix>.json).
# Batch 4 (406M peaks ~12GB): more steps on a small dataset. That's too much for
# the 3060 next to the API, so the Beekeeper project's 20GB minimum puts it on the 3090.
train_loader, val_loader = make_loaders(mix, cfg, batch_size=4)

# Fresh optimizer, LR well below pretraining's 6e-4 peak: adapt the format
# without overwriting what pretraining learned.
model = LanguageModel(gpt_config=cfg,
                      train_loader=train_loader,
                      val_loader=val_loader,
                      checkpoint_path=output_checkpoint,
                      lr=1e-4)
model.model.load_state_dict(torch.load(init_checkpoint, map_location=model.device))
print(f"Loaded pretrained weights from {init_checkpoint}; mix {mix}; saving to {output_checkpoint}")

# The number to beat: the pretrained model's assistant-token loss on the same val set.
model.model.eval()
with torch.no_grad():
    print(f"Val loss before post-training: {model.calc_loss_loader(val_loader):.3f}")
model.model.train()

# ~4.9M tokens per epoch. Two epochs: on run 25, epoch 3 moved val loss only
# 2.548 -> 2.543 while train loss kept falling (run 29). The best epoch is saved.
model.train(num_epochs=2,
            eval_freq=200,
            eval_iter=20,
            start_context="### Question\nWhat is the difference between a list and a tuple in Python?\n\n### Answer\n",
            patience=1,
            warmup_steps=100)
