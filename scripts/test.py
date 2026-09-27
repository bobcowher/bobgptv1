import sys
from pathlib import Path

# Make the project root importable when run as scripts/<name>.py
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch

from chat_template import ROLE_HEADERS
from config import GPT_CONFIG_124M
from languagemodel import LanguageModel


def qa_prompt(question):
    # Open an answer turn in the training chat format and let the model fill it in.
    return f"{ROLE_HEADERS['user']}\n{question}\n\n{ROLE_HEADERS['assistant']}\n"


PROMPTS = {
    "prose": [
        "Every effort moves you",
    ],
    "code": [
        "def fibonacci(n):\n",
        "import json\n\n\ndef load_config(path):\n",
        "class Stack:\n    \"\"\"A simple LIFO stack.\"\"\"\n\n    def __init__(self):\n",
        "for i, line in enumerate(",
    ],
    "qa": [
        qa_prompt("What is the difference between a list and a tuple in Python?"),
        qa_prompt("How do I read a file line by line in Python?"),
        qa_prompt("What does the `yield` keyword do?"),
    ],
}


model = LanguageModel(gpt_config=GPT_CONFIG_124M)
model.load_the_model()
model.model.eval()

context_size = model.model.pos_emb.weight.shape[0]
eot = model.tokenizer.eot_token

for kind, prompts in PROMPTS.items():
    for prompt in prompts:
        encoded = model.text_to_token_ids(prompt, model.tokenizer).to(model.device)
        with torch.no_grad():
            token_ids = model.generate(
                    idx=encoded,
                    max_new_tokens=120, context_size=context_size,
                    temperature=0.8, top_k=40, eos_id=eot
                    )
        completion = model.token_ids_to_text(token_ids[:, encoded.shape[1]:], model.tokenizer)
        print(f"===== {kind} " + "=" * 50)
        print(prompt, end="")
        print(f"\033[36m{completion}\033[0m")
        print()
