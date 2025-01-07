import os
import requests
import tiktoken
import numpy as np
from pathlib import Path


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
enc = tiktoken.get_encoding("gpt2")
train_ids = enc.encode_ordinary(train_data)
val_ids = enc.encode_ordinary(val_data)
print(f"train has {len(train_ids):,} tokens")
print(f"val has {len(val_ids):,} tokens")

# export to bin files
train_ids = np.array(train_ids, dtype=np.uint16)
val_ids = np.array(val_ids, dtype=np.uint16)
train_ids.tofile(os.path.join(os.path.dirname(__file__), 'train.bin'))
val_ids.tofile(os.path.join(os.path.dirname(__file__), 'val.bin'))
