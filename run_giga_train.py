from datetime import datetime
import time
import logging
import argparse
from pathlib import Path
from tqdm import tqdm
import os

import torch
from torch.utils.data import Dataset, DataLoader

from seq2seq.utils import load_bert
from seq2seq.multiTokenizer import loadBertTokenizer

from rouge import Rouge  # pip install rouge
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction

class BertDataset(Dataset):
    """
    针对特定数据集，定义一个相关的取数据的方式
    """

    def __init__(self, sents_src, sents_tgt, tokenizer, prefix):
        ## 一般init函数是加载所有数据
        super(BertDataset, self).__init__()
        # 读原始数据
        # self.sents_src, self.sents_tgt = read_corpus(poem_corpus_dir)
        self.sents_src = sents_src
        self.sents_tgt = sents_tgt

        self.idx2word = {k: v for v, k in tokenizer.vocab.items()}
        self.tokenizer = tokenizer
        self.prefix = prefix
    def __getitem__(self, i):
        ## 得到单个数据
        # print(i)
        src = self.prefix+self.sents_src[i]
        tgt = self.prefix+self.sents_tgt[i]
        out = self.tokenizer(src, tgt)
        output = {
            "token_ids": out['input_ids'],
            "token_type_ids": out['token_type_ids']
        }
        return output

    def __len__(self):
        return len(self.sents_src)


def load_corpus(args):
    """
    读原始数据
    """
    sents_src = []
    sents_tgt = []
    in_path = str(Path(args.data_dir).joinpath(args.src_file))
    out_path = str(Path(args.data_dir).joinpath(args.tgt_file))
    with open(in_path, "r", encoding="utf-8") as f:
        lines = f.readlines()
        for line in lines:
            sents_src.append(line.strip())
    with open(out_path, "r", encoding="utf-8") as f:
        lines = f.readlines()
        for line in lines:
            sents_tgt.append(line.strip())


    return sents_src, sents_tgt

def load_valid_corpus(args):
    """
    read valid data
    """
    sents_src = []
    sents_tgt = []
    in_path = str(Path(args.data_dir).joinpath(args.valid_src_file))
    out_path = str(Path(args.data_dir).joinpath(args.valid_tgt_file))
    with open(in_path, "r", encoding="utf-8") as f:
        lines = f.readlines()
        for line in lines:
            sents_src.append(line.strip())
    with open(out_path, "r", encoding="utf-8") as f:
        lines = f.readlines()
        for line in lines:
            sents_tgt.append(line.strip())

    return [*zip(sents_src,sents_src)]



    return sents_src, sents_tgt

def collate_fn(batch):
    """
    动态padding， batch为一部分sample
    """

    def padding(indice, max_length, pad_idx=0):
        """
        pad 函数
        """
        pad_indice = [item + [pad_idx] * max(0, max_length - len(item)) for item in indice]
        return torch.tensor(pad_indice)

    token_ids = [data["token_ids"] for data in batch]
    max_length = max([len(t) for t in token_ids])
    token_type_ids = [data["token_type_ids"] for data in batch]

    token_ids_padded = padding(token_ids, max_length)
    token_type_ids_padded = padding(token_type_ids, max_length)
    target_ids_padded = token_ids_padded[:, 1:].contiguous()

    return token_ids_padded, token_type_ids_padded, target_ids_padded

def evaluate(model, data, tokenizer, logger, topk = 1):
    rouge = Rouge()
    smooth = SmoothingFunction().method1
    best_bleu = 0.
    def run_eval():
        total = 0
        rouge_1, rouge_2, rouge_l, bleu = 0, 0, 0, 0
        for content, title in tqdm(data):
            title = ' '.join(tokenizer.decode(tokenizer.encode(title)).split()[1:-1])
            pred_title = ' '.join(model.generate(content, beam_size=topk))
            #todo add batch sampling shit
            if pred_title.strip():
                scores = rouge.get_scores(hyps=pred_title, refs=title)
                rouge_1 += scores[0]['rouge-1']['f']
                rouge_2 += scores[0]['rouge-2']['f']
                rouge_l += scores[0]['rouge-l']['f']
                bleu += sentence_bleu(
                    references=[title.split(' ')],
                    hypothesis=pred_title.split(' '),
                    smoothing_function = smooth
                )
        rouge_1 /= total
        rouge_2 /= total
        rouge_l /= total
        bleu /= total
        return {
            'rouge-1': rouge_1,
            'rouge-2': rouge_2,
            'rouge-l': rouge_l,
            'bleu': bleu,
        }
    metrics = run_eval()
    if metrics['bleu'] > best_bleu:
        best_bleu = metrics['bleu']
        model.save_weights('./best_model.weights')
        logger.info('valid_data:', metrics)



def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--data_dir",
                        default=None,
                        type=str,
                        required=True,
                        help="The input data dir. Should contain the .tsv files (or other data files) for the task.")
    parser.add_argument("--src_file", default=None, type=str,required=True,
                        help="The input data file name.")
    parser.add_argument("--tgt_file", default=None, type=str,required=True,
                        help="The output data file name.")
    parser.add_argument("--valid_src_file", default=None, type=str,required=True,
                        help="The valid src")
    parser.add_argument("--valid_tgt_file", default=None, type=str,required=True,
                        help="The valid tgt")

    parser.add_argument("--bert_model", default=None, type=str, required=True,
                        help="Bert pre-trained model selected in the list: bert-base-uncased, "
                             "bert-large-uncased, bert-base-cased, bert-base-multilingual, bert-base-chinese.")
    parser.add_argument("--config_path", default=None, type=str,
                        help="Bert config file path.")
    parser.add_argument("--vocab_path", default=None, type=str,required=True,
                        help="vocab file path.")
    parser.add_argument("--output_dir",
                        default=None,
                        type=str,
                        required=True,
                        help="The output directory where the model predictions and checkpoints will be written.")
    parser.add_argument("--log_dir",
                        default='',
                        type=str,
                        required=True,
                        help="The output directory where the log will be written.")
    parser.add_argument("--model_recover_path",
                        default=None,
                        type=str,
                        required=True,
                        help="The file of fine-tuned pretraining model.")

    # parser.add_argument("--optim_recover_path",
    #                     default=None,
    #                     type=str,
    #                     help="The file of pretraining optimizer.")


    #other
    parser.add_argument("--do_train",
                        action='store_true',
                        help="Whether to run training.")
    parser.add_argument("--do_eval",
                        action='store_true',
                        help="Whether to run eval on the dev set.")
    parser.add_argument("--train_batch_size",
                        default=32,
                        required=True,
                        type=int,
                        help="Total batch size for training.")
    parser.add_argument("--eval_batch_size",
                        default=64,
                        type=int,
                        help="Total batch size for eval.")
    parser.add_argument("--learning_rate", default=5e-5, type=float,
                        help="The initial learning rate for Adam.")
    parser.add_argument("--weight_decay",
                        default=0.01,
                        type=float,
                        help="The weight decay rate for Adam.")
    parser.add_argument("--num_train_epochs",
                        default=3,
                        type=int,
                        help="Total number of training epochs to perform.")
    parser.add_argument("--warmup_proportion",
                        default=0.1,
                        type=float,
                        help="Proportion of training to perform linear learning rate warmup for. "
                             "E.g., 0.1 = 10%% of training.")

    # parser.add_argument("--no_cuda",
    #                     action='store_true',
    #                     help="Whether not to use CUDA when available")


    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(args.log_dir, exist_ok=True)

    assert Path(args.model_recover_path).exists(
    ), "--model_recover_path doesn't exist"


    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(args.log_dir, exist_ok=True)

    #set logger
    logger = logging.getLogger(__file__)
    logger.setLevel(level=logging.INFO)
    now = datetime.now()
    experiment_time = now.strftime("%H_%M_%S")
    logfilename = experiment_time+"log.txt"
    logflepath = str(Path(args.log_dir).joinpath(logfilename))
    handler = logging.FileHandler(logflepath)
    handler.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    logger.addHandler(handler)
    logger.addHandler(console)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    tokenizer = loadBertTokenizer(args.vocab_path)

    #prepare data
    sents_src, sents_tgt = load_corpus(args)
    sents_src = sents_src[:100]
    sents_tgt = sents_tgt[:100]
    valid_pair = load_valid_corpus(args)


    logger.info("device: " + str(device))
    model_name = "bert"
    bert_model = load_bert(tokenizer,model_name = model_name, model_class='seq2seq')
    bert_model.load_pretrain_params(args.model_recover_path)
    bert_model.set_device(device)
    optim_parameters = list(bert_model.parameters())
    optimizer = torch.optim.Adam(optim_parameters, lr=args.learning_rate, weight_decay=args.weight_decay)
    giga_dataset = BertDataset(sents_src, sents_tgt,tokenizer,prefix='[_giga]')
    giga_dataloader = DataLoader(giga_dataset, batch_size=args.train_batch_size, shuffle=True, collate_fn=collate_fn)


    def train(epoch):
        # 一个epoch的训练
        bert_model.train()
        iteration(epoch, dataloader=giga_dataloader, train=True)
    def save(save_path):
        """
        保存模型
        """
        bert_model.save_all_params(save_path)
        logger.info("{} saved!".format(save_path))

    def iteration(epoch, dataloader, train=True):
        total_loss = 0
        start_time = time.time()
        step = 0
        for token_ids, token_type_ids, target_ids in tqdm(dataloader, position=0, leave=True):
            step += 1
            # if step % 5000 == 0:
            #     # bert_model.eval()
            #     # test_data = [
            #     #     "police arrested five anti-nuclear protesters thursday after they sought to disrupt loading of a french antarctic research and supply vessel , a spokesman for the protesters said ."]
            #     # for text in test_data:
            #     #     logger.info(bert_model.generate(text, beam_size=3))
            #     # bert_model.train()
            #     pass

            predictions, loss = bert_model(token_ids,
                                        token_type_ids,
                                        labels=target_ids,
                                        )
            if train:
                # 清空之前的梯度
                optimizer.zero_grad()
                # 反向传播, 获取新的梯度
                loss.backward()
                # 用获取的梯度更新模型参数
                optimizer.step()

            # 为计算当前epoch的平均loss
        total_loss += loss.item()
        end_time = time.time()
        spend_time = end_time - start_time

        logger.info("epoch is " + str(epoch) + ". loss is " + str(total_loss) + ". spend time is " + str(spend_time))
            # 保存模型
        checkpoint_name = experiment_time+"epoch"+f"{epoch}"+".bin"

        # save(str(Path(args.output_dir).joinpath(checkpoint_name)))
        evaluate(bert_model,valid_pair,tokenizer,logger,topk=1)

    for epoch in range(args.num_train_epochs):
        # 训练一个epoch
        train(epoch)

if __name__ == '__main__':
    main()












