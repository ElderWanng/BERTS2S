from transformers import BertTokenizer
tokenizer = BertTokenizer.from_pretrained('../models/BertBase/vocab.txt')
ids = tokenizer.encode("I never thought I could do that!")
# print(tokenizer.encode("I never thought I could do that!"))
# print(tokenizer.decode(ids))
#
# print(tokenizer.encode("This is the official PyTorch package ","for the discrete VAE used for DALL·E."))
ids2 = tokenizer.encode("This is the official PyTorch package ","for the discrete VAE used for DALL·E.")
print("--------------------")
# print(tokenizer.decode(ids2))
print(tokenizer("This is the official PyTorch package ","for the discrete VAE used for DALL·E."))