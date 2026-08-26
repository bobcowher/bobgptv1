import re

class SimpleTokenizerV1:

    def __init__(self, vocab):
        self.str_to_int = vocab
        self.int_to_str = {i:s for s,i in vocab.items()}

    def encode(self, text):
        preprocessed = re.split(r'([,.:;?_!"()\']|--|\s)', text)
        preprocessed = [item for item in preprocessed if item.strip()]
        idx = [self.str_to_int.get(token) for token in preprocessed]
        return idx

    def decode(self, ids):
        tokens = [self.int_to_str.get(id) for id in ids]
        text = " ".join(tokens)
        text = re.sub(r'\s+([,.?!"()\'])', r'\1', text)

        return text 
