import random
import shutil
import warnings
from datetime import datetime
import time
import logging
import argparse
from pathlib import Path
from torch.backends import cudnn
from tqdm import tqdm
import os
import torch
from torch.utils import data
from seq2seq.utils import load_bert
from seq2seq.multiTokenizer import loadBertTokenizer

from rouge import Rouge  # pip install rouge
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
from torch.utils.data import Dataset, DataLoader
import torch.distributed as dist
import torch.multiprocessing as mp




class BertDataset(Dataset):
    """
    针对特定数据集，定义一个相关的取数据的方式
    """

    def __init__(self, sents_src, sents_tgt, tokenizer):
        ## 一般init函数是加载所有数据
        super(BertDataset, self).__init__()
        # 读原始数据
        # self.sents_src, self.sents_tgt = read_corpus(poem_corpus_dir)
        self.sents_src = sents_src
        self.sents_tgt = sents_tgt

        self.idx2word = {k: v for v, k in tokenizer.vocab.items()}
        self.tokenizer = tokenizer
    def __getitem__(self, i):
        ## 得到单个数据
        # print(i)
        src = self.sents_src[i]
        tgt = self.sents_tgt[i]
        out = self.tokenizer(src, tgt)
        output = {
            "token_ids": out['input_ids'],
            "token_type_ids": out['token_type_ids']
        }
        return output

    def __len__(self):
        return len(self.sents_src)
class valid_dataset(data.Dataset):
    def __init__(self,src,tgt):
        self.src = src
        self.tgt = tgt

    def __getitem__(self,index):
        content = self.src[index]
        title = self.tgt[index]
        return content,title

    def __len__(self):
        return len(self.src)

def load_corpus(args):
    """
    读原始数据
    """
    sents_src = []
    sents_tgt = []
    in_path = str(Path(args.data_dir).joinpath(args.src_file1))
    out_path = str(Path(args.data_dir).joinpath(args.tgt_file1))
    print("here")
    with open(in_path, "r", encoding="utf-8") as f:
        lines = f.readlines()
        for line in lines:
            sents_src.append(line.strip())
    with open(out_path, "r", encoding="utf-8") as f:
        lines = f.readlines()
        for line in lines:
            sents_tgt.append(line.strip())
    print("end")
    return sents_src, sents_tgt

def load_valid_corpus(args):
    """
    read valid data
    """
    sents_src = []
    sents_tgt = []
    in_path = str(Path(args.data_dir).joinpath(args.valid_src_file1))
    out_path = str(Path(args.data_dir).joinpath(args.valid_tgt_file1))
    print("here")
    with open(in_path, "r", encoding="utf-8") as f:
        lines = f.readlines()
        for line in lines:
            sents_src.append(line.strip())
    with open(out_path, "r", encoding="utf-8") as f:
        lines = f.readlines()
        for line in lines:
            sents_tgt.append(line.strip())
    print("end")
    return sents_src,sents_tgt

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

# def evaluate(model, valid_loader, tokenizer,args, logger):
#     src = [
#         "japan 's nec corp. and UNK computer corp. of the united states said wednesday they had agreed to join forces in supercomputer sales .",
#         "the sri lankan government on wednesday announced the closure of government schools with immediate effect as a military campaign against tamil separatists escalated in the north of the country .",
#         "police arrested five anti-nuclear protesters thursday after they sought to disrupt loading of a french antarctic research and supply vessel , a spokesman for the protesters said ."]
#     tg = ["nec UNK in computer sales tie-up", "sri lanka closes schools as war escalates",
#           "protesters target french research ship"]
#     with torch.no_grad():
#         model.eval()
#         titles = [' '.join(tokenizer.decode(tokenizer.encode(title)).split()[1:-1]) for title in src]
#         pred_titles = model.multiTask_batch_generate(titles)
#         for i in range(len(pred_titles)):
#             print(src[i])
#             print(tg[i])
#             print(pred_titles[i])
#             print('____________________________________')
#
#     pred_titles = model.multiTask_batch_generate(titles)
#     for i in pred_titles:
#         print(i)
#     rouge = Rouge()
#     smooth = SmoothingFunction().method1

def reduce_mean(tensor, nprocs):
    rt = tensor.clone()
    dist.all_reduce(rt, op=dist.ReduceOp.SUM)
    rt /= nprocs
    return rt

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)

class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print('\t'.join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--data_dir",
                        default=None,
                        type=str,
                        required=True,
                        help="The input data dir. Should contain the .tsv files (or other data files) for the task.")
    parser.add_argument("--src_file1", default=None, type=str,required=True,
                        help="The input data file name.")
    parser.add_argument("--tgt_file1", default=None, type=str,required=True,
                        help="The output data file name.")
    parser.add_argument("--valid_src_file1", default=None, type=str,required=True,
                        help="The valid src")
    parser.add_argument("--valid_tgt_file1", default=None, type=str,required=True,
                        help="The valid tgt")





    parser.add_argument("--bert_model", default=None, type=str, required=True,
                        help="Bert pre-trained model selected in the list: bert-base-uncased, "
                             "bert-large-uncased, bert-base-cased, bert-base-multilingual, bert-base-chinese.")
    parser.add_argument("--config_path", default=None, type=str,
                        help="Bert config file path.")
    parser.add_argument("--vocab_path", default=None, type=str,required=True,
                        help="vocab file path.")
    parser.add_argument("--model_out_path",
                        default=None,
                        type=str,
                        required=True,
                        help="The output path where the model predictions and checkpoints will be written.")
    parser.add_argument("--log_dir",
                        default='',
                        type=str,
                        required=True,
                        help="The output directory where the log will be written.")
    parser.add_argument("--model_recover_path",
                        default=None,
                        type=str,
                        required=False,
                        help="The file of fine-tuned pretraining model.")

    #other
    parser.add_argument("--do_train",
                        action='store_true',
                        help="Whether to run training.")
    parser.add_argument("--train_batch_size",
                        default=32,
                        required=True,
                        type=int,
                        help="Total batch size for training.")
    parser.add_argument("--eval_batch_size",
                        default=128,
                        type=int,
                        help="Total batch size for eval.")
    parser.add_argument("--learning_rate", default=1e-5, type=float,
                        help="The initial learning rate for Adam.")
    parser.add_argument('--momentum', default=0.9, type=float, metavar='M', help='momentum')
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

    parser.add_argument("--cut_data",
                        default=0,
                        type=int,
                        help="keep a part of data for fast debug")
    parser.add_argument('--seed', default=2, type=int, help='seed for initializing training. ')
    parser.add_argument('--local_rank', default=-1, type=int,
                        help='node rank for distributed training')
    parser.add_argument('-p', '--print_freq', default=10, type=int, metavar='N', help='print frequency (default: 10)')

    args = parser.parse_args()
    args.nprocs = torch.cuda.device_count()
    # args.nprocs = 1

    os.makedirs(args.log_dir, exist_ok=True)
    out_path = Path(args.model_out_path).absolute()
    os.makedirs(out_path.parent, exist_ok=True)
    # assert Path(args.model_recover_path).exists(
    # ), "--model_recover_path doesn't exist"

    mp.spawn(main_worker, nprocs=args.nprocs, args=(args.nprocs, args))

    #set logger
    # logger = logging.getLogger(__file__)
    # logger.setLevel(level=logging.INFO)
    #
    # now = datetime.now()
    # experiment_time = now.strftime("%H_%M_%S")
    # logfilename = experiment_time+"log.txt"
    # logflepath = str(Path(args.log_dir).joinpath(logfilename))
    # handler = logging.FileHandler(logflepath)
    # handler.setLevel(logging.INFO)
    # formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    # handler.setFormatter(formatter)
    # console = logging.StreamHandler()
    # console.setLevel(logging.INFO)
    # logger.addHandler(handler)
    # logger.addHandler(console)
    #
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # # special_dict = {'additional_special_tokens': []}
    # # special_dict[f"additional_special_tokens"].append(args.prefix1)
    # # if not args.single_mode:
    # #     special_dict[f"additional_special_tokens"].append(args.aux_prefix)
    #
    #
    # tokenizer = loadBertTokenizer(args.vocab_path)
    #
    #
    # sents_src, sents_tgt = load_corpus(args)
    # valid_src, valid_tgt = load_valid_corpus(args)
    # if(args.cut_data>0):
    #     cut_num = args.cut_data
    #     sents_src = sents_src[:cut_num]
    #     sents_tgt = sents_tgt[:cut_num]
    #     valid_src = valid_src[:cut_num]
    #     valid_tgt = valid_tgt[:cut_num]
    # giga_dataset = BertDataset(sents_src, sents_tgt,tokenizer)
    # train_sampler = torch.utils.data.distributed.DistributedSampler(giga_dataset)
    # giga_dataloader = DataLoader(giga_dataset, batch_size=args.train_batch_size, shuffle=False, collate_fn=collate_fn,sampler=train_sampler)
    #
    # logger.info("device: " + str(device))
    #
    # best_bleu = 0
    #
    # model_name = "bert"
    # bert_model = load_bert(tokenizer,model_name = model_name, model_class='seq2seq')
    # if args.model_recover_path and Path(args.model_recover_path).exists():
    #     bert_model.load_pretrain_params(args.model_recover_path)
    # #
    # # bert_model.load_pretrain_params(args.model_recover_path)
    # bert_model.set_device(device)
    #
    # bert_model = torch.nn.parallel.DistributedDataParallel(bert_model, device_ids=[args.local_rank],find_unused_parameters=True)
    # optim_parameters = list(bert_model.parameters())
    # optimizer = torch.optim.Adam(optim_parameters, lr=args.learning_rate, weight_decay=args.weight_decay)
    # torch.cuda.set_device(args.local_rank)
    # task1_valid_dataset = valid_dataset(valid_src, valid_tgt)
    # valid_loader = data.DataLoader(task1_valid_dataset,batch_size=16,shuffle=False)


    # def train(epoch):
    #     # 一个epoch的训练
    #     bert_model.train()
    #     iteration(epoch, train_dataloader=giga_dataloader,valid_loader=valid_loader ,train=True)
    # def save(save_path):
    #     """
    #     保存模型
    #     """
    #     bert_model.save_all_params(save_path)
    #     logger.info("{} saved!".format(save_path))

    # def iteration(epoch, train_dataloader, valid_loader, train=True):
    #     total_loss = 0
    #     start_time = time.time()
    #     step = 0
    #
    #     for token_ids, token_type_ids, target_ids in tqdm(train_dataloader, position=0, leave=True):
    #         bert_model.train()
    #         token_ids = token_ids.cuda(non_blocking=True)
    #         token_type_ids = token_type_ids.cuda(non_blocking=True)
    #         target_ids = target_ids.cuda(non_blocking=True)
    #         step += 1
    #
    #         predictions, loss = bert_model(token_ids,
    #                                     token_type_ids,
    #                                     labels=target_ids,
    #                                     )
    #         # loss = loss.mean()
    #         # print(loss)
    #
    #         if train:
    #             optimizer.zero_grad()
    #             loss.backward()
    #             optimizer.step()
    #         if train and step % 1000 ==0:
    #             scoope_out(bert_model)
    #         # 为计算当前epoch的平均loss
    #         total_loss += loss.item()
    #
    #     scoope_out(bert_model)
    #     end_time = time.time()
    #     spend_time = end_time - start_time
    #
    #     logger.info("epoch is " + str(epoch) + ". loss is " + str(total_loss) + ". spend time is " + str(spend_time))
def scoope_out(bert_model, tokenizer):
    with torch.no_grad():
        bert_model.eval()
        src = [
            "japan 's nec corp. and UNK computer corp. of the united states said wednesday they had agreed to join forces in supercomputer sales .",
            "the sri lankan government on wednesday announced the closure of government schools with immediate effect as a military campaign against tamil separatists escalated in the north of the country .",
            "police arrested five anti-nuclear protesters thursday after they sought to disrupt loading of a french antarctic research and supply vessel , a spokesman for the protesters said ."]
        tg = ["nec UNK in computer sales tie-up", "sri lanka closes schools as war escalates",
              "protesters target french research ship"]
        with torch.no_grad():
            bert_model.eval()
            titles = [' '.join(tokenizer.decode(tokenizer.encode(title)).split()[1:-1]) for title in src]
            # pred_titles = bert_model.modules().multiTask_batch_generate(titles,args.prefix1)
            pred_titles = bert_model.module.multiTask_batch_generate(titles)
            for i in range(len(pred_titles)):
                print(src[i])
                print(tg[i])
                print(pred_titles[i])
                print('____________________________________')


def main_worker(local_rank, nprocs, args):
    print(local_rank)
    args.local_rank = local_rank
    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        cudnn.deterministic = True
        warnings.warn('You have chosen to seed training. '
                      'This will turn on the CUDNN deterministic setting, '
                      'which can slow down your training considerably! '
                      'You may see unexpected behavior when restarting '
                      'from checkpoints.')

    best_bleu = 0.0
    # dist.init_process_group(backend='gloo', init_method="file:///C:/Users/tianshu/cache.txt", world_size=args.nprocs,
    #                         rank=local_rank)

    dist.init_process_group(backend='nccl', init_method='tcp://127.0.0.1:23456', world_size=args.nprocs, rank=local_rank)
    # print("here", local_rank)
    tokenizer = loadBertTokenizer(args.vocab_path)
    # print("here", local_rank)
    sents_src, sents_tgt = load_corpus(args)
    valid_src, valid_tgt = load_valid_corpus(args)
    print("here", local_rank)
    if (args.cut_data > 0):
        cut_num = args.cut_data
        sents_src = sents_src[:cut_num]
        sents_tgt = sents_tgt[:cut_num]
        valid_src = valid_src[:cut_num]
        valid_tgt = valid_tgt[:cut_num]
    train_dataset = BertDataset(sents_src, sents_tgt, tokenizer)
    val_dataset = valid_dataset(valid_src, valid_tgt)

    train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)

    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size=args.train_batch_size,
                                               num_workers=2,
                                               pin_memory=True,
                                               collate_fn=collate_fn,
                                               sampler=train_sampler)
    val_loader = torch.utils.data.DataLoader( val_dataset,
                                               batch_size=32
                                               )
    # if args.evaluate:
    #     validate(val_loader, model, criterion, local_rank, args)
    #     return
    model_name = "bert"
    # print("here", local_rank)
    model = load_bert(tokenizer, model_name=model_name, model_class='seq2seq')
    # print("here", local_rank)
    model.set_device(device = torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    if args.model_recover_path and Path(args.model_recover_path).exists():
        model.load_pretrain_params(args.model_recover_path)
    torch.cuda.set_device(local_rank)
    model.cuda(local_rank)
    args.train_batch_size = int(args.train_batch_size / args.nprocs)
    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[local_rank],find_unused_parameters=True)
    print(model.device)
    optimizer = torch.optim.SGD(model.parameters(), args.learning_rate, momentum=args.momentum, weight_decay=args.weight_decay)
    cudnn.benchmark = True
    for epoch in range(args.num_train_epochs):
        train_sampler.set_epoch(epoch)
        # val_sampler.set_epoch(epoch)
        #todo adjust_learning_rate
        # adjust_learning_rate(optimizer, epoch, args)
        train(train_loader, model, optimizer, epoch, local_rank, args)
        bleu = validate(val_loader, model,tokenizer, local_rank, args)['bleu']
        is_best = bleu > best_bleu
        best_bleu = max(bleu, best_bleu)
        if args.local_rank == 0:
            save_checkpoint(
                {
                    'epoch': epoch + 1,
                    'state_dict': model.module.state_dict(),
                    'best_bleu': best_bleu,
                }, is_best)

def train(train_loader, model, optimizer, epoch, local_rank, args):
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    progress = ProgressMeter(len(train_loader), [batch_time, data_time, losses],
                             prefix="Epoch: [{}]".format(epoch))

    # switch to train mode
    model.train()

    end = time.time()
    for i, (token_ids, token_type_ids, target_ids) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)

        token_ids = token_ids.cuda(local_rank, non_blocking=True)
        token_type_ids = token_type_ids.cuda(local_rank, non_blocking=True)
        target_ids = target_ids.cuda(local_rank, non_blocking=True)
        batch_size = token_ids.size(0)
        # compute output
        predictions, loss = model(token_ids,
                                       token_type_ids,
                                       labels=target_ids,
                                       )
        torch.distributed.barrier()
        torch.distributed.barrier()
        reduced_loss = reduce_mean(loss, args.nprocs)
        losses.update(reduced_loss.item(), batch_size)

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            progress.display(i)

def validate(val_loader, model,tokenizer, local_rank, args):
    model.eval()
    # switch to evaluate mode
    src = [
        "japan 's nec corp. and UNK computer corp. of the united states said wednesday they had agreed to join forces in supercomputer sales .",
        "the sri lankan government on wednesday announced the closure of government schools with immediate effect as a military campaign against tamil separatists escalated in the north of the country .",
        "police arrested five anti-nuclear protesters thursday after they sought to disrupt loading of a french antarctic research and supply vessel , a spokesman for the protesters said ."]
    tg = ["nec UNK in computer sales tie-up", "sri lanka closes schools as war escalates",
          "protesters target french research ship"]
    with torch.no_grad():
        model.eval()
        titles = [' '.join(tokenizer.decode(tokenizer.encode(title)).split()[1:-1]) for title in src]
        pred_titles = model.module.multiTask_batch_generate(titles)
        for i in range(len(pred_titles)):
            print(src[i])
            print(tg[i])
            print(pred_titles[i])
            print('____________________________________')

    pred_titles = model.module.multiTask_batch_generate(titles)
    for i in pred_titles:
        print(i)
    rouge = Rouge()
    smooth = SmoothingFunction().method1

    def run_eval():
        total = 0
        rouge_1, rouge_2, rouge_l, bleu = 0, 0, 0, 0
        for batch in tqdm(val_loader):
            contents = list(batch[0])
            titles = list(batch[1])
            total += len(titles)
            titles = [' '.join(tokenizer.decode(tokenizer.encode(title)).split()[1:-1]) for title in titles]
            # title = ' '.join(tokenizer.decode(tokenizer.encode(title)).split()[1:-1])
            pred_titles = model.module.multiTask_batch_generate(contents)
            # todo add batch sampling shit
            for i in range(len(titles)):
                pred_title = pred_titles[i]
                title = titles[i]
                if pred_title.strip():
                    scores = rouge.get_scores(hyps=pred_title, refs=title)
                    rouge_1 += scores[0]['rouge-1']['f']
                    rouge_2 += scores[0]['rouge-2']['f']
                    rouge_l += scores[0]['rouge-l']['f']
                    bleu += sentence_bleu(
                        references=[title.split(' ')],
                        hypothesis=pred_title.split(' '),
                        smoothing_function=smooth
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
    return metrics


def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, 'model_best.pth.tar')




if __name__ == '__main__':
    main()












