from importlib.metadata import version
from dataset import *
import tiktoken
import torch
from attention import *
from models import GPTModel, TransformerBlock, generate_text_simple
from config import GPT_CONFIG_124M 

tokenizer = tiktoken.get_encoding("gpt2")

torch.manual_seed(123)

model = GPTModel(cfg=GPT_CONFIG_124M)

start_context = "Hello, I am"
encoded = tokenizer.encode(start_context)
print("encoded:", encoded)
encoded_tensor = torch.tensor(encoded).unsqueeze(0)
print("encoded_tensor.shape:", encoded_tensor.shape)

model.eval()

out = generate_text_simple(
        model=model,
        idx=encoded_tensor,
        max_new_tokens=6,
        context_size=GPT_CONFIG_124M["context_length"]
        )

print(out)

decoded = tokenizer.decode(out.squeeze(0).tolist())

print(decoded)
