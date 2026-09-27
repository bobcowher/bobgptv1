import sys
from pathlib import Path

# Make the project root importable when run as scripts/<name>.py
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dataset import *
from config import GPT_CONFIG_124M 
from languagemodel import LanguageModel


# A mix in mixes/, built into data/build/ by scripts/build_mix.py
train_loader, val_loader = make_loaders("pretrain_v2", GPT_CONFIG_124M, batch_size=8)


# ~215M train tokens: ~4.5h/epoch on the 3090. Early stopping usually ends it sooner.
num_epochs = 5

model = LanguageModel(gpt_config=GPT_CONFIG_124M, 
                      train_loader=train_loader, 
                      val_loader=val_loader)

model.train(num_epochs=num_epochs,
            eval_freq=1000,
            eval_iter=20,
            start_context="Every effort moves you",
            patience=2)
