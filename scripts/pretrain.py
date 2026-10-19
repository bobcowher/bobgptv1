import os
import sys
from pathlib import Path

# Make the project root importable when run as scripts/<name>.py
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import GPT_CONFIG_406M
from dataset import SwitchLoader, make_loaders
from languagemodel import LanguageModel, wsd_decay_start

# No dropout: in one pass over fresh data nothing repeats, so there is nothing to
# overfit and dropout only slows learning. Post-training (small data) keeps 0.1.
cfg = {**GPT_CONFIG_406M, "drop_rate": 0.0}

# WSD on a fixed token budget: MIX for the stable phase, DECAY_MIX for the decay,
# as the tuning runs chose. ~2.8B tokens is ~40h on the 3090; a pipeline smoke
# test sets PRETRAIN_TOKENS small.
mix = os.environ.get("MIX", "pretrain_v14_se16")
decay_mix = os.environ.get("DECAY_MIX", "pretrain_decay_v3")
max_steps = int(float(os.environ.get("PRETRAIN_TOKENS", "2.8e9"))) // (8 * cfg["context_length"])

# Val is v11's, the set run 31 and every tuning run were measured on.
_, val_loader = make_loaders("pretrain_v11", cfg, batch_size=8)  # 406M at batch 8 peaks ~18 GiB
train_loader, _ = make_loaders(mix, cfg, batch_size=8)
decay_loader, _ = make_loaders(decay_mix, cfg, batch_size=8)
train_loader = SwitchLoader(train_loader, decay_loader, wsd_decay_start(max_steps))

# checkpoints/ is a Beekeeper output path, kept in the run's own directory;
# a pipeline's later steps get a copy of it with inherit: [checkpoints].
model = LanguageModel(gpt_config=cfg,
                      train_loader=train_loader,
                      val_loader=val_loader,
                      checkpoint_path="checkpoints/pretrain/model.pth",
                      lr=3e-4,  # 6e-4 suits 124M; GPT-3 used 3e-4 at 350M
                      compile=True)  # a minute of compiling is nothing on a multi-hour run
print(f"Pretrain: {max_steps} steps; mix {mix} then {decay_mix}")

model.train(num_epochs=1,
            eval_freq=1000,
            eval_iter=20,
            start_context="Every effort moves you",
            max_steps=max_steps,
            lr_schedule="wsd",
            save_each_eval=True)  # don't lose hours to a crash
