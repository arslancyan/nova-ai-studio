"""Small deterministic tokenizer for NOVA-0.

This is intentionally simple: the first model needs a reproducible text
conditioning path, not a huge language model.
"""

import string

PAD = 0
UNK = 1
CHARS = " " + string.ascii_letters + string.digits + string.punctuation
VOCAB = ["<pad>", "<unk>"] + list(CHARS)
STOI = {ch: i + 2 for i, ch in enumerate(CHARS)}
ITOS = {i: ch for ch, i in STOI.items()}

def encode(text: str, max_length: int = 96):
    ids = [STOI.get(ch, UNK) for ch in text[:max_length]]
    ids += [PAD] * (max_length - len(ids))
    return ids

def decode(ids):
    return "".join(ITOS.get(int(i), "") for i in ids if int(i) not in (PAD, UNK))

def vocab_size():
    return len(VOCAB)
