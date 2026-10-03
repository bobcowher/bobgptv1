import os
import sys
from pathlib import Path

import torch

# Make the project root importable when run as scripts/<name>.py
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import GPT_CONFIG_124M
from languagemodel import LanguageModel, PRETRAIN_CHECKPOINT, POSTTRAIN_CHECKPOINT


# Post-training starts from the pretrained weights. Fail loudly if they're
# missing or don't fit: fine-tuning a random model would look like a run.
if not os.path.exists(PRETRAIN_CHECKPOINT):
    sys.exit(f"No pretrained checkpoint at {PRETRAIN_CHECKPOINT}; run scripts/pretrain.py first.")

model = LanguageModel(gpt_config=GPT_CONFIG_124M, checkpoint_path=POSTTRAIN_CHECKPOINT)
model.model.load_state_dict(torch.load(PRETRAIN_CHECKPOINT, map_location=model.device))
print(f"Loaded pretrained weights from {PRETRAIN_CHECKPOINT}")

# TODO: Q&A + chat data with loss on assistant tokens only, lower LR, fresh optimizer.
sys.exit("posttrain.py: training loop not written yet.")
