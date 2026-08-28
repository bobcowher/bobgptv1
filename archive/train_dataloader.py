from importlib.metadata import version
from dataset import *
import tiktoken
import torch

vocab_size = 50257
output_dim = 256
max_length = 4
batch_size = 8

token_embedding_layer = torch.nn.Embedding(vocab_size, output_dim)
pos_embedding_layer   = torch.nn.Embedding(vocab_size, output_dim) 

tokenizer = tiktoken.get_encoding("gpt2")

with open("data/the-verdict.txt", "r", encoding="utf-8") as f:
    raw_text = f.read()

dataloader = create_dataloader_v1(raw_text, 
                                  batch_size=batch_size, 
                                  max_length=max_length,
                                  stride=1, 
                                  shuffle=False)

data_iter = iter(dataloader)

inputs, targets = next(data_iter)

token_embeddings = token_embedding_layer(inputs)
pos_embeddings = pos_embedding_layer(inputs)

print(token_embeddings.shape)
print(pos_embeddings.shape)

input_embeddings = token_embeddings + pos_embeddings

print(input_embeddings.shape)
