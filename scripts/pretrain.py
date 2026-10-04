import sys
from pathlib import Path

# Make the project root importable when run as scripts/<name>.py
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dataset import *
from config import GPT_CONFIG_406M
from languagemodel import LanguageModel, PRETRAIN_CHECKPOINT

# No dropout: in one pass over fresh data nothing repeats, so there is nothing to
# overfit and dropout only slows learning. Post-training (small data) keeps 0.1.
cfg = {**GPT_CONFIG_406M, "drop_rate": 0.0}

# A mix in mixes/, built into data/build/ by scripts/build_mix.py
train_loader, val_loader = make_loaders("pretrain_v11", cfg, batch_size=8)  # 406M at batch 8 peaks ~18 GiB


# One pass over v11 (3.0B tokens, ~40h at an estimated ~21k tok/s on the 260W 3090):
# fresh data beats repeats.
# The LR schedule decays to its floor at the end of the last epoch, so
# num_epochs is the training budget, not a cap.
num_epochs = 1

# Saved under data/ (the shared dataset dir on lab) so posttrain.py, which runs
# as a separate Beekeeper project, can load it.
model = LanguageModel(gpt_config=cfg, 
                      train_loader=train_loader, 
                      val_loader=val_loader,
                      checkpoint_path=PRETRAIN_CHECKPOINT,
                      lr=3e-4,  # 6e-4 suits 124M; GPT-3 used 3e-4 at 350M
                      compile=True)  # a minute of compiling is nothing on a multi-hour run

model.train(num_epochs=num_epochs,
            eval_freq=1000,
            eval_iter=20,
            start_context="Every effort moves you",
            patience=2,
            save_each_eval=True)
