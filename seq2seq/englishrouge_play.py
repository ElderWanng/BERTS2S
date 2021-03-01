from rouge import Rouge
from transformers import BertTokenizer
tokenizer = BertTokenizer.from_pretrained("bert-cased-base")
hypothesis = "18-year-old"

rouge = Rouge()
scores = rouge.get_scores(hypothesis, reference)
print(scores)