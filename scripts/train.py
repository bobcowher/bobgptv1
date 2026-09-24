import sys
from pathlib import Path

# Make the project root importable when run as scripts/<name>.py
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dataset import *
from config import GPT_CONFIG_124M 
from languagemodel import LanguageModel


file_path = Path("data/pretrain/combined_corpus.txt")
text_data = file_path.read_text(encoding="utf-8")

train_loader, val_loader = make_loaders(text_data, GPT_CONFIG_124M, batch_size=16)
del text_data


num_epochs = 10

model = LanguageModel(gpt_config=GPT_CONFIG_124M, 
                      train_loader=train_loader, 
                      val_loader=val_loader)

model.train(num_epochs=num_epochs,
            eval_freq=1000,
            eval_iter=20,
            start_context="Every effort moves you")
