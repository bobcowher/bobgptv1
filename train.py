from importlib.metadata import version
from dataset import *
import tiktoken
import torch
from attention import SelfAttention_v1

inputs = torch.tensor(
        [[0.43, 0.15, 0.89],
         [0.55, 0.87, 0.66],
         [0.57, 0.85, 0.64],
         [0.22, 0.58, 0.33],
         [0.77, 0.25, 0.10],
         [0.05, 0.80, 0.55]]
        )

query = inputs[1]

attention = SelfAttention_v1(d_in=6, d_out=6)

# attn_scores = torch.empty(6, 6)
#
# for i, x_i in enumerate(inputs):
#     for j, x_j in enumerate(inputs):
#         attn_scores[i, j] = torch.dot(x_i, x_j) 
#
# print(attn_scores)

# attn_scores = inputs @ inputs.T
#
# attn_weights = torch.softmax(attn_scores, dim=-1)
#
# print(attn_scores)
# print(attn_weights)
#

# x_2   = inputs[1]
# d_in  = inputs.shape[1]
# d_out = 2
#
# torch.manual_seed(123)
#
# W_query = torch.nn.Parameter(torch.rand(d_in, d_out), requires_grad=False)
# W_key   = torch.nn.Parameter(torch.rand(d_in, d_out), requires_grad=False)
# W_value = torch.nn.Parameter(torch.rand(d_in, d_out), requires_grad=False)
#
# query_2 = x_2 @ W_query
# key_2   = x_2 @ W_key
# value_2 = x_2 @ W_value
#
# print(query_2)
# print(key_2)
# print(value_2)
