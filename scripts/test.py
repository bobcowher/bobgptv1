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
        "fn lower_bound(values: &[i32], target: i32) -> usize {\n",
        "std::vector<int> stable_unique(std::span<const int> values) {\n",
    ],
    "qa": [
        qa_prompt("What is the difference between a list and a tuple in Python?"),
        qa_prompt("How do I read a file line by line in Python?"),
        qa_prompt("What does the `yield` keyword do?"),
        qa_prompt("Why does Rust prevent a vector from being mutated while one of its elements is borrowed?"),
        qa_prompt("When should C++ code use std::unique_ptr instead of std::shared_ptr?"),
        qa_prompt("What is the difference between Linux permitted and effective capability sets?"),
    ],
    "chat": [
        qa_prompt("hi! who are you?"),
        qa_prompt("What's today's date?"),
        qa_prompt("thanks, that helped"),
        qa_prompt("Why is the sky blue?"),
        qa_prompt("Rewrite this to sound more polite: send me the report now."),
        qa_prompt("I have an exam tomorrow and I'm really nervous."),
    ],
}


# python scripts/test.py [checkpoint]   (default checkpoints/model.pth)
# Seeded, so two checkpoints get the same sampling draws and can be compared side by side.
torch.manual_seed(0)
model = LanguageModel(gpt_config=GPT_CONFIG_124M)
model.load_the_model(sys.argv[1] if len(sys.argv) > 1 else None)
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
