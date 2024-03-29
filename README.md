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
on my mac
```shell
python run_giga_train_aux.py
--data_dir
/Users/mac/Downloads/ggw_data/
--bert_model
bert
--output_dir
./.outs
--log_dir
./.logs
--model_recover_path
/Users/mac/PycharmProjects/BERTS2S/models/chinese_wwm_ext_pytorch/pytorch_model.bin
--vocab_path
/Users/mac/PycharmProjects/BERTS2S/models/chinese_wwm_ext_pytorch/vocab.txt
--src_file1
train.src.10k
--tgt_file1
train.tgt.10k
--train_batch_size
1
--valid_src_file1
test.src
--valid_tgt_file1
test.tgt
--single_mode
False
--prefix1
[_giga]
```

multitask code
```shell
--data_dir
/Users/mac/Downloads/ggw_data/
--bert_model
bert
--output_dir
./.outs
--log_dir
./.logs
--model_recover_path
/Users/mac/PycharmProjects/BERTS2S/models/chinese_wwm_ext_pytorch/pytorch_model.bin
--vocab_path
/Users/mac/PycharmProjects/BERTS2S/models/chinese_wwm_ext_pytorch/vocab.txt
--src_file1
train.src.10k
--tgt_file1
train.tgt.10k
--train_batch_size
1
--valid_src_file1
test.src
--valid_tgt_file1
test.tgt
--single_mode
False
--prefix1
[_giga]
--aux_data_dir
/Users/mac/Downloads/multinli_1.0
--aux_data_dir
/Users/mac/Downloads/multinli_1.0
--aux_src_file
train.src.txt
--aux_tgt_file
train.tgt.txt
--aux_valid_src_file
dev.src.txt
--aux_valid_tgt_file
dev.tgt.txt
--aux_prefix
[_aux]
```
```shell
--data_dir
/content/drive/MyDrive/corpus/org_data
--bert_model
bert
--output_dir
./.outs
--log_dir
./.logs
--model_recover_path
/content/drive/MyDrive/BertBase/model.bin
--vocab_path
/content/drive/MyDrive/BertBase/vocab.txt
--src_file1
train.src.txt
--tgt_file1
train.tgt.txt
--train_batch_size
16
--valid_src_file1
test.src.txt
--valid_tgt_file1
test.tgt.txt
--single_mode
False
--prefix1 
[_giga]
--aux_data_dir
/content/drive/MyDrive/corpus/multiNLI
--aux_src_file
train.src.txt
--aux_tgt_file
train.tgt.txt
--aux_valid_src_file
dev.src.txt
--aux_valid_tgt_file
dev.tgt.txt
--aux_prefix
[_aux]
```
## Slurm code:
```shell
srun --cpus-per-task=4 --time=2:00:00 --mem=4000 --gres=gpu:2 --pty /bin/bash
singularity exec --nv --overlay $SCRATCH/overlay-50G-10M.ext3:rw /scratch/work/public/singularity/cuda10.1-cudnn7-devel-ubuntu18.04-20201207.sif /bin/bash

source /ext3/env.sh
conda activate
```
```shell
python run_giga_train_aux.py  --data_dir  /home/tw2112/datas/data_upload/giga  --bert_model  bert  --model_out_path  ./outs/best.bin  --log_dir  ./.logs  --vocab_path  models/bert_cased_base/vocab.txt  --src_file1  train.src.txt  --tgt_file1  train.tgt.txt  --train_batch_size  64  --valid_src_file1  test.src.txt  --valid_tgt_file1  test.tgt.txt  --single_mode  False  --prefix1  [_giga]  --aux_data_dir  /home/tw2112/datas/data_upload/multinli  --aux_src_file  train.src.txt  --aux_tgt_file  train.tgt.txt  --aux_valid_src_file  dev.src.txt  --aux_valid_tgt_file  dev.tgt.txt  --aux_prefix  [_aux]  --cut_data  10000 --num_train_epochs 10
```

## multi GPU with DDP
```shell
CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch --nproc_per_node=1 train_single_multiGPU.py --data_dir  /home/tw2112/datas/data_upload/giga  --bert_model  bert  --model_out_path  ./models/best.bin  --log_dir  ./logs  --vocab_path  /home/tw2112/codes/project1/models/bertbase/vocab.txt  --src_file1  train.src.txt  --tgt_file1  train.tgt.txt  --train_batch_size  64  --valid_src_file1  test.src.txt  --valid_tgt_file1  test.tgt.txt 
```

```shell
 python train_single_multiGPU.py --data_dir  /home/tw2112/datas/data_upload/giga  --bert_model  bert  --model_out_path  ./models/best.bin  --log_dir  ./logs  --vocab_path  /home/tw2112/codes/project1/models/bertbase/vocab.txt  --src_file1  train.src.txt  --tgt_file1  train.tgt.txt  --train_batch_size  64  --valid_src_file1  test.src.txt  --valid_tgt_file1  test.tgt.txt --model_recover_path  /home/tw2112/codes/project1/models/bertbase/model.bin --cut_data 100
```