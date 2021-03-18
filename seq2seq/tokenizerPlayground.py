from typing import List

import numpy
import torch
from transformers import BertTokenizer
tokenizer = BertTokenizer.from_pretrained("bert-base-cased")
source = "18-year-old"
# This is the official PyTorch package for the discrete VAE used for DALL·E.


# print(tokenizer.batch_decode(res['input_ids']))
texts = ["pytorch is God","tensorflow is dust"]
for i in range(len(texts)):
    texts[i] = "[_giga]"+texts[i]
res = tokenizer.batch_encode_plus(texts)
# print(res)
# print(tokenizer.tokenize("[_giga]"))
task_prefix_tokens = tokenizer("[_giga]",add_special_tokens=False)["input_ids"]
def add_tokens_to_batch_idx(batch_token,new_tokens:List[List]):
    assert len(batch_token)==len(new_tokens)
    for i in range(len(batch_token)):
        batch_token[i] = batch_token[i] + new_tokens[i]
res = tokenizer(texts)
print(tokenizer.vocab_size)
add_tokens_to_batch_idx(res["input_ids"],[task_prefix_tokens for _ in range(2)])
print(res)
print(tokenizer.batch_decode(res['input_ids']))
i = tokenizer.add_special_tokens({'additional_special_tokens':["[_giga]"]})
print(i)
print(tokenizer.vocab_size)
tokenizer.add_tokens(["[_giga]]","[_aux]"])
print(tokenizer.vocab_size)
texts = ["pytorch is God","tensorflow is dust"]
for i in range(len(texts)):
    texts[i] = "[_giga]"+texts[i]
res = tokenizer.batch_encode_plus(texts)
print(tokenizer.batch_decode(res['input_ids']))
