import re
from tokenizer import SimpleTokenizerV1

with open("data/the-verdict.txt", "r", encoding="utf8") as f:
    raw_text = f.read()

print("Total number of character:", len(raw_text))

preprocessed = re.split(r'([,.:;?_!"()\']|--|\s)', raw_text)
preprocessed = [item for item in preprocessed if item.strip()]

all_words = sorted(set(preprocessed))
all_words.extend(["<|endoftext|>", "<|unk|>"])
vocab_size = len(all_words)

vocab = {token:integer for integer,token in enumerate(all_words)}

tokenizer = SimpleTokenizerV1(vocab=vocab) 

# text = ["My", "life"]

text = "It's not my time I'm not going"

idx = tokenizer.encode(text)

print(idx)

tokens = tokenizer.decode(idx)

print(tokens)

