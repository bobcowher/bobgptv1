from importlib.metadata import version
from dataset import *
import tiktoken
import torch
from attention import *
from models import GPTModel, TransformerBlock 

GPT_CONFIG_124M = {
        "vocab_size": 50257,
        "context_length": 1024,
        "emb_dim": 768,
        "n_heads": 12,
        "n_layers": 12,
        "drop_rate": 0.1,
        "qkv_bias": False
        }


tokenizer = tiktoken.get_encoding("gpt2")
batch = []

txt1 = "Every effort moves you"
txt2 = "Every day holds a"

batch.append(torch.tensor(tokenizer.encode(txt1)))
batch.append(torch.tensor(tokenizer.encode(txt2)))
batch = torch.stack(batch, dim=0)

torch.manual_seed(123)

model = GPTModel(cfg=GPT_CONFIG_124M)

out = model(batch)
print("Input batch:\n", batch)
print("\nOutput shape:", out.shape)

total_params = sum(p.numel() for p in model.parameters())
print(f"Total number of parameters: {total_params:,}")




# torch.manual_seed(123)
# x = torch.rand(2, 4, 768)
# block = TransformerBlock(GPT_CONFIG_124M)
# output = block(x)
#
# print("Input shape: ", x.shape)
# print("Output shape: ", output.shape)
#

