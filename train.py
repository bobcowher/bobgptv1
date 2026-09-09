from dataset import *
from config import GPT_CONFIG_124M 
from languagemodel import LanguageModel


file_path = "data/the-verdict.txt"
with open(file_path, "r", encoding="utf-8") as file:
    text_data = file.read()

train_loader, val_loader = make_loaders(text_data, GPT_CONFIG_124M)


num_epochs = 50 

model = LanguageModel(gpt_config=GPT_CONFIG_124M, 
                      train_loader=train_loader, 
                      val_loader=val_loader)

model.train(num_epochs=num_epochs,
            eval_freq=5,
            eval_iter=5,
            start_context="Every effort moves you")



