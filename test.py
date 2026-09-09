from importlib.metadata import version
from dataset import *
import tiktoken
import torch
from models import *
from config import GPT_CONFIG_124M 

tokenizer = tiktoken.get_encoding("gpt2")

torch.manual_seed(123)

model = GPTModel(cfg=GPT_CONFIG_124M)
model.eval()

start_context = "Every effort moves you"

token_ids = generate_text_simple(
        model=model,
        idx=text_to_token_ids(start_context, tokenizer),
        max_new_tokens=10,
        context_size=GPT_CONFIG_124M["context_length"]
        )

print("Output text:\n", token_ids_to_text(token_ids, tokenizer))
