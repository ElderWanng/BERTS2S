torch==1.7.1
transformers==4.3.3
tqdm

parameters

```
python run_giga_train.py
--data_dir
D:\codeproject\NLP\data\ggw_data
--bert_model
bert
--output_dir
./.outs
--log_dir
./.logs
--model_recover_path
C:\Users\tianshu\PycharmProjects\bert_s2s\models\bertbase\pytorch_model.bin
--vocab_path
C:\Users\tianshu\PycharmProjects\bert_s2s\models\bertbase\vocab.txt
--src_file
train.src.10k
--tgt_file
train.tgt.10k
--train_batch_size
8
```