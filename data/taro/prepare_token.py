import os
import pickle
import requests
import tiktoken
import numpy as np
from pathlib import Path
from transformers import AutoTokenizer


current_file = Path(__file__)
two_levels_up = current_file.parents[3]

input_files_path = os.path.join(two_levels_up, 'data/txt')
input_file_paths = os.listdir(input_files_path)
input_file_paths = [os.path.join(input_files_path, path) for path in input_file_paths]

data = ""
for path in input_file_paths:
    with open(path, "r", encoding='windows-1251') as file:
        content = file.read()
    data = data + '\n' + content

n = len(data)
train_data = data[:int(n*0.9)]
val_data = data[int(n*0.9):]

# encode with tiktoken gpt2 bpe
# enc = tiktoken.get_encoding("gpt2")
# train_ids = enc.encode_ordinary(train_data)
# val_ids = enc.encode_ordinary(val_data)

tokenizer = AutoTokenizer.from_pretrained("Dmitriy007/rugpt2_gen_news")
train_ids = tokenizer(train_data)['input_ids']
val_ids = tokenizer(val_data)['input_ids']
print(f"train has {len(train_ids):,} tokens")
print(f"val has {len(val_ids):,} tokens")

# export to bin files
train_ids = np.array(train_ids, dtype=np.uint16)
val_ids = np.array(val_ids, dtype=np.uint16)
train_ids.tofile(os.path.join(os.path.dirname(__file__), 'train.bin'))
val_ids.tofile(os.path.join(os.path.dirname(__file__), 'val.bin'))

# save the meta information as well, to help us encode/decode later
vocab_dict = tokenizer.get_vocab()
vocab_size = len(vocab_dict)
itos = {idx: tokenizer.decode([idx]) for token, idx in vocab_dict.items()}
stoi = {token: idx for token, idx in vocab_dict.items()}

meta = {
    'vocab_size': vocab_size,
    'itos': itos,
    'stoi': stoi,
}

with open(os.path.join(os.path.dirname(__file__), 'meta.pkl'), 'wb') as f:
    pickle.dump(meta, f)
