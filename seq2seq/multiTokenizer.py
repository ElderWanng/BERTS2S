from transformers import BertTokenizer
def loadBertTokenizer(path):
    tokenizer = BertTokenizer.from_pretrained(path)
    return tokenizer
