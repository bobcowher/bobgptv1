from importlib.metadata import version
from dataset import *
import tiktoken
import torch
from attention import *
from models import *
from config import GPT_CONFIG_124M 

tokenizer = tiktoken.get_encoding("gpt2")

torch.manual_seed(123)

model = GPTModel(cfg=GPT_CONFIG_124M)
model.eval()

start_context = "Every effort moves you"

inputs = torch.tensor([[16833, 3626, 6100],
                        [40, 1107, 588]])
targets = torch.tensor([[3626, 6100, 345 ],
                        [1107, 588, 11311]])

with torch.no_grad():
    logits = model(inputs)

print(logits.shape)

probas = torch.softmax(logits, dim=-1)
print(probas.shape)

token_ids = torch.argmax(probas, dim=-1, keepdim=True)
print(token_ids.shape)
print(token_ids)

print(f"Targets batch 1: {token_ids_to_text(targets[0], tokenizer)}")
print(f"Outputs batch 1: {token_ids_to_text(token_ids[0].flatten(), tokenizer)}")

text_idx = 0
target_probas_1 = probas[text_idx, [0, 1, 2], targets[text_idx]]
print("Text 1:", target_probas_1)

text_idx = 1
target_probas_2 = probas[text_idx, [0, 1, 2], targets[text_idx]]
print("Text 2:", target_probas_2)

log_probas = torch.log(torch.cat((target_probas_1, target_probas_2)))
print(log_probas)

avg_log_probas = torch.mean(log_probas)
print(avg_log_probas)

# token_ids = generate_text_simple(
#         model=model,
#         idx=text_to_token_ids(start_context, tokenizer),
#         max_new_tokens=10,
#         context_size=GPT_CONFIG_124M["context_length"]
#         )
#
# print("Output text:\n", token_ids_to_text(token_ids, tokenizer))
