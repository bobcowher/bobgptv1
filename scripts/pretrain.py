import sys
from pathlib import Path

# Make the project root importable when run as scripts/<name>.py
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dataset import *
from config import GPT_CONFIG_124M 
from languagemodel import LanguageModel, PRETRAIN_CHECKPOINT


# A mix in mixes/, built into data/build/ by scripts/build_mix.py
train_loader, val_loader = make_loaders("pretrain_v10", GPT_CONFIG_124M, batch_size=8)


# One pass over ~1.3B tokens (~12h at 30k tok/s): fresh data beats repeats.
# The LR schedule decays to its floor at the end of the last epoch, so
# num_epochs is the training budget, not a cap.
num_epochs = 1

# Saved under data/ (the shared dataset dir on lab) so posttrain.py, which runs
# as a separate Beekeeper project, can load it.
model = LanguageModel(gpt_config=GPT_CONFIG_124M, 
                      train_loader=train_loader, 
                      val_loader=val_loader,
                      checkpoint_path=PRETRAIN_CHECKPOINT)

model.train(num_epochs=num_epochs,
            eval_freq=1000,
            eval_iter=20,
            start_context="Every effort moves you",
            patience=2,
            save_each_eval=True)
