from importlib.metadata import version
from dataset import *
import tiktoken
import torch
from attention import *

inputs = torch.tensor(
        [[0.43, 0.15, 0.89],
         [0.55, 0.87, 0.66],
         [0.57, 0.85, 0.64],
         [0.22, 0.58, 0.33],
         [0.77, 0.25, 0.10],
         [0.05, 0.80, 0.55]]
        )

query = inputs[1]
d_in = inputs.shape[1]
d_out = 2

# print(d_in)
# print(d_out)

batch = torch.stack((inputs, inputs), dim=0)

context_length = batch.shape[1]

# attention = CausalAttention(d_in=d_in, 
#                             d_out=d_out,
#                             context_length=context_length,
#                             dropout=0.1)

attention = MultiHeadAttentionWrapper(d_in=d_in,
                                      d_out=d_out,
                                      context_length=context_length,
                                      dropout=0.1,
                                      num_heads=2)

context_vec = attention(batch)

# print(query)
print(context_vec)
print(context_vec.shape)

