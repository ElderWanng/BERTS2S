from transformers import BertTokenizer
tokenizer = BertTokenizer.from_pretrained("bert-base-cased")
source = "18-year-old"
# This is the official PyTorch package for the discrete VAE used for DALL·E.
ids2 = tokenizer.encode(source)
print(ids2)
print("--------------------")
print(source)
print(tokenizer.decode(ids2))
print(' '.join(tokenizer.decode(ids2).split()[1:-1]))