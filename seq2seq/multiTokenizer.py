from transformers import BertTokenizer
# from tokenizers import BertWordPieceTokenizer as BertTokenizer
def loadBertTokenizer(path,special_dict={}):
    tokenizer = BertTokenizer(path)
    tokenizer.add_special_tokens(special_dict)
    return tokenizer
