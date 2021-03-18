from typing import List, Optional

import torch
import torch.nn as nn 
import torch.nn.functional as F

import time
# from tokenizers import BertWordPieceTokenizer as BertTokenizer
from transformers import BertTokenizer

from seq2seq.config import yayun_list
import os 
from seq2seq.basic_bert import BasicBert

class Seq2SeqModel(BasicBert):
    """
    """
    def __init__(self, tokenizer, model_name="roberta"):
        super(Seq2SeqModel, self).__init__()
        self.tokenizer:BertTokenizer = tokenizer
        config = ""
        if model_name == "roberta":
            from seq2seq.model.roberta_model import BertModel, BertConfig, BertLMPredictionHead
            config = BertConfig(len(tokenizer))
            self.bert = BertModel(config)
            self.decoder = BertLMPredictionHead(config, self.bert.embeddings.word_embeddings.weight)
        elif model_name == "bert":
            from seq2seq.model.bert_model import BertConfig, BertModel, BertLMPredictionHead
            config = BertConfig(len(tokenizer))
            self.bert = BertModel(config)
            self.decoder = BertLMPredictionHead(config, self.bert.embeddings.word_embeddings.weight)
        else :
            raise Exception("model_name_err")
            
        self.hidden_dim = config.hidden_size
        self.vocab_size = tokenizer.vocab_size
        self.word2ix= tokenizer.vocab


    def compute_loss(self, predictions, labels, target_mask):
        """
        target_mask : 句子a部分和pad部分全为0， 而句子b部分为1
        """
        predictions = predictions.view(-1, len(self.tokenizer))
        labels = labels.view(-1)
        target_mask = target_mask.view(-1).float()
        loss = nn.CrossEntropyLoss(ignore_index=0, reduction="none")
        return (loss(predictions, labels) * target_mask).sum() / target_mask.sum() ## 通过mask 取消 pad 和句子a部分预测的影响
    
    def forward(self, input_tensor, token_type_id, position_enc=None, labels=None):
        ## 传入输入，位置编码，token type id ，还有句子a 和句子b的长度，注意都是传入一个batch数据
        ##  传入的几个值，在seq2seq 的batch iter 函数里面都可以返回
        input_tensor = input_tensor.to(self.device)
        token_type_id = token_type_id.to(self.device)
        if position_enc is not None:
            position_enc = position_enc.to(self.device)
        if labels is not None :
            labels = labels.to(self.device)
        input_shape = input_tensor.shape
        batch_size = input_shape[0]
        seq_len = input_shape[1]
        ## 构建特殊的mask
        ones = torch.ones((1, 1, seq_len, seq_len), dtype=torch.float32, device=self.device)
        a_mask = ones.tril() # 下三角矩阵
        s_ex12 = token_type_id.unsqueeze(1).unsqueeze(2).float()
        s_ex13 = token_type_id.unsqueeze(1).unsqueeze(3).float()
        a_mask = (1.0 - s_ex12) * (1.0 - s_ex13) + s_ex13 * a_mask 
            
        enc_layers, _ = self.bert(input_tensor, position_ids=position_enc, token_type_ids=token_type_id, attention_mask=a_mask, 
                                    output_all_encoded_layers=True)
        squence_out = enc_layers[-1] ## 取出来最后一层输出

        predictions = self.decoder(squence_out)

        if labels is not None:
            ## 计算loss
            ## 需要构建特殊的输出mask 才能计算正确的loss
            # 预测的值不用取最后sep符号的结果 因此是到-1
            predictions = predictions[:, :-1].contiguous()
            target_mask = token_type_id[:, 1:].contiguous()
            loss = self.compute_loss(predictions, labels, target_mask)
            return predictions, loss 
        else :
            return predictions
    
    def generate(self, text, out_max_length=40, beam_size=1, is_poem=False, max_length=256):
        # 对 一个 句子生成相应的结果
        ## 通过输出最大长度得到输入的最大长度，这里问题不大，如果超过最大长度会进行截断
        self.out_max_length = out_max_length
        input_max_length = max_length - out_max_length
        # print(text)
        res = self.tokenizer(text, max_length=input_max_length)
        token_ids, token_type_ids = res['input_ids'],res['token_type_ids'];
        token_ids = torch.tensor(token_ids, device=self.device).view(1, -1)
        token_type_ids = torch.tensor(token_type_ids, device=self.device).view(1, -1)
        if is_poem:## 古诗的beam-search稍有不同
            
            out_puts_ids = self.beam_search_poem(text, token_ids, token_type_ids, self.word2ix, beam_size=beam_size, device=self.device)
        else :   
            out_puts_ids = self.beam_search(token_ids, token_type_ids, self.word2ix, beam_size=beam_size, device=self.device)
        
        # 解码 得到相应输出
        # if err is False:
        #     return self.tokenizer.decode(out_puts_ids)
        
        return self.tokenizer.decode(out_puts_ids.cpu().numpy())


    def beam_search(self, token_ids, token_type_ids, word2ix, beam_size=1, device="cpu"):
        """
        beam-search操作
        """
        sep_id = word2ix["[SEP]"]
        
        # 用来保存输出序列
        output_ids = torch.empty(1, 0, device=device, dtype=torch.long)
        # 用来保存累计得分
      
        with torch.no_grad(): 
            output_scores = torch.zeros(token_ids.shape[0], device=device)
            for step in range(self.out_max_length):
                if step == 0:
                    scores = self.forward(token_ids, token_type_ids)
                    # 重复beam-size次 输入ids
                    token_ids = token_ids.view(1, -1).repeat(beam_size, 1)
                    token_type_ids = token_type_ids.view(1, -1).repeat(beam_size, 1)
                else:
                    scores = self.forward(new_input_ids, new_token_type_ids)
                
                logit_score = torch.log_softmax(scores[:, -1], dim=-1)
                
                logit_score = output_scores.view(-1, 1) + logit_score # 累计得分
                ## 取topk的时候我们是展平了然后再去调用topk函数
                # 展平
                logit_score = logit_score.view(-1)
                hype_score, hype_pos = torch.topk(logit_score, beam_size)
                indice1 = (hype_pos // scores.shape[-1]) # 行索引
                indice2 = (hype_pos % scores.shape[-1]).long().reshape(-1, 1) # 列索引
               
                # 更新得分
                output_scores = hype_score
                output_ids = torch.cat([output_ids[indice1], indice2], dim=1).long()
                new_input_ids = torch.cat([token_ids, output_ids], dim=1)
                new_token_type_ids = torch.cat([token_type_ids, torch.ones_like(output_ids)], dim=1)

                end_counts = (output_ids == sep_id).sum(1)  # 统计出现的end标记
                best_one = output_scores.argmax()
                if end_counts[best_one] == 1:
                    # 说明出现终止了～
                    return output_ids[best_one][:-1]
                else :
                    # 保留未完成部分
                    flag = (end_counts < 1)  # 标记未完成序列
                    if not flag.all():  # 如果有已完成的
                        token_ids = token_ids[flag]
                        token_type_ids = token_type_ids[flag]
                        new_input_ids = new_input_ids[flag]
                        new_token_type_ids = new_token_type_ids[flag]
                        output_ids = output_ids[flag]  # 扔掉已完成序列
                        output_scores = output_scores[flag]  # 扔掉已完成序列
                        end_counts = end_counts[flag]  # 扔掉已完成end计数
                        beam_size = flag.sum()  # topk相应变化
    
            return output_ids[output_scores.argmax()]

    def beam_search_poem(self, text, token_ids, token_type_ids, word2ix, beam_size=1, device="cpu"):
        """
        beam-search操作
        """
        yayun_pos = []
        title = text.split("##")[0]
        if "五言律诗" in text:
            yayun_pos = [10, 22, 34, 46]
        elif "五言绝句" in text:
            yayun_pos = [10, 22]
        elif "七言律诗" in text:
            yayun_pos = [14, 30, 46, 62]
        elif "七言绝句" in text:
            yayun_pos = [14, 30]
        sep_id = word2ix["[SEP]"]
        douhao_id = word2ix["，"]# 逗号
        ix2word = {v: k for k, v in word2ix.items()}
        juhao_id = word2ix["。"]# 句号
        repeat_word = [[] for i in range(beam_size)]
        # 用来保存输出序列
        output_ids = torch.empty(1, 0, device=device, dtype=torch.long)
        last_chars = torch.empty(1, 0, device=device, dtype=torch.long)
        yayun_chars = (-1) * torch.ones(beam_size, dtype=torch.long)
        start = 0
        with torch.no_grad(): 
            output_scores = torch.zeros(token_ids.shape[0], device=device)
            for step in range(self.out_max_length):
                if step == 0:
                    scores = self.forward(token_ids, token_type_ids)
                    # 重复beam-size次 输入ids
                    token_ids = token_ids.view(1, -1).repeat(beam_size, 1)
                    token_type_ids = token_type_ids.view(1, -1).repeat(beam_size, 1)
                else:
                    scores = self.forward(new_input_ids, new_token_type_ids)
                
                logit_score = torch.log_softmax(scores[:, -1], dim=-1)
                
                for i, char in enumerate(last_chars):
                    
                    for word in repeat_word[i]:
                        logit_score[i, word] -= 5
                    for word in title:
                        ix = word2ix.get(word, -1)
                        if ix != -1:
                            logit_score[i, ix] += 2

                if step in yayun_pos:
                    # print("step is " + str(step))
                    # print("yayun_chars is " + str(yayun_chars))
                    for i, char in enumerate(last_chars):
                        if yayun_chars[i].item() != -1:
                            yayuns = yayun_list[yayun_chars[i].item()]
                            for char in yayuns:
                                ix = word2ix.get(char, -1)
                                if ix != -1:
                                    # print("char is " + str(char))
                                    logit_score[i, ix] += 10


                logit_score = output_scores.view(-1, 1) + logit_score # 累计得分
                ## 取topk的时候我们是展平了然后再去调用topk函数
                # 展平
                logit_score = logit_score.view(-1)
                hype_score, hype_pos = torch.topk(logit_score, beam_size)
                indice1 = (hype_pos // scores.shape[-1]) # 行索引
                indice2 = (hype_pos % scores.shape[-1]).long().reshape(-1, 1) # 列索引
                
                for index, each_out in zip(indice1, indice2):
                    index = index.item()
                    each_out = each_out.item()
                    
                    if each_out in repeat_word[index]:
                        pass 
                        # repeat_word[index].append(each_out)
                        # hype_score[index] -= 2 * repeat_word[index].count(each_out)
                    else :
                        repeat_word[index].append(each_out)
                    
                    if start < beam_size and each_out == douhao_id and len(last_chars) != 0:
                        start += 1
                        word = ix2word[last_chars[index].item()]# 找到上一个字符 记住其押韵情况
                        for i, each_yayun in enumerate(yayun_list):
                            if word in each_yayun:
                                yayun_chars[index] = i
                                break

                # 更新得分
                output_scores = hype_score

                last_chars = indice2

                output_ids = torch.cat([output_ids[indice1], indice2], dim=1).long()
                new_input_ids = torch.cat([token_ids, output_ids], dim=1)
                new_token_type_ids = torch.cat([token_type_ids, torch.ones_like(output_ids)], dim=1)

                end_counts = (output_ids == sep_id).sum(1)  # 统计出现的end标记
                best_one = output_scores.argmax()
                if end_counts[best_one] == 1:
                    # 说明出现终止了～
                    # print(repeat_word)
                    # print(yayun_chars)
                    return output_ids[best_one][:-1]
                else :
                    # 保留未完成部分
                    flag = (end_counts < 1)  # 标记未完成序列
                    if not flag.all():  # 如果有已完成的
                        token_ids = token_ids[flag]
                        token_type_ids = token_type_ids[flag]
                        last_chars = last_chars[flag]
                        yayun_chars = yayun_chars[flag]
                        new_input_ids = new_input_ids[flag]
                        new_token_type_ids = new_token_type_ids[flag]
                        output_ids = output_ids[flag]  # 扔掉已完成序列
                        output_scores = output_scores[flag]  # 扔掉已完成序列
                        end_counts = end_counts[flag]  # 扔掉已完成end计数
                        beam_size = flag.sum()  # topk相应变化
                        flag = flag.long()

                        new_repeat_word = []
                        for index, i in enumerate(flag):
                            if i.item() == 1:
                                new_repeat_word.append(repeat_word[index])
                     
                        repeat_word = new_repeat_word

            return output_ids[output_scores.argmax()]
    
    def beam_search_poem_v2(self, text, token_ids, token_type_ids, word2ix, beam_size=1, device="cpu"):
        """
        beam-search操作
        """
        yayun_pos = []
        if "五言律诗" in text:
            yayun_pos = [10, 22, 34, 46]
        elif "五言绝句" in text:
            yayun_pos = [10, 22]
        elif "七言律诗" in text:
            yayun_pos = [14, 30, 46, 62]
        elif "七言绝句" in text:
            yayun_pos = [14, 30]
        sep_id = word2ix["[SEP]"]
        douhao_id = word2ix["，"]# 逗号
        ix2word = {v: k for k, v in word2ix.items()}
        juhao_id = word2ix["。"]# 句号
        repeat_word = []
        # 用来保存输出序列
        output_ids = torch.empty(1, 0, device=device, dtype=torch.long)
        last_chars = torch.empty(1, 0, device=device, dtype=torch.long)
        yayun_chars = (-1) * torch.ones(beam_size, dtype=torch.long)
        start = 0
        with torch.no_grad(): 
            output_scores = torch.zeros(token_ids.shape[0], device=device)
            for step in range(self.out_max_length):
                if step == 0:
                    scores = self.forward(token_ids, token_type_ids)
                    # 重复beam-size次 输入ids
                    token_ids = token_ids.view(1, -1).repeat(beam_size, 1)
                    token_type_ids = token_type_ids.view(1, -1).repeat(beam_size, 1)
                else:
                    scores = self.forward(new_input_ids, new_token_type_ids)
                
                logit_score = torch.log_softmax(scores[:, -1], dim=-1)
                # if len(last_chars) != 0:
                #     logit_score[last_chars] -= 5
                for i, char in enumerate(last_chars):
                    logit_score[i, char] -= 2
                    for word in repeat_word:
                        logit_score[i, word] -= 1
                if step in yayun_pos:
                    # print("step is " + str(step))
                    # print("yayun_chars is " + str(yayun_chars))
                    for i, char in enumerate(last_chars):
                        if yayun_chars[i].item() != -1:
                            yayuns = yayun_list[yayun_chars[i].item()]
                            for char in yayuns:
                                ix = word2ix.get(char, -1)
                                if ix != -1:
                                    # print("char is " + str(char))
                                    logit_score[i, ix] += 3
                logit_score = output_scores.view(-1, 1) + logit_score # 累计得分
                ## 取topk的时候我们是展平了然后再去调用topk函数
                # 展平
                logit_score = logit_score.view(-1)
                hype_score, hype_pos = torch.topk(logit_score, beam_size)
                indice1 = (hype_pos // scores.shape[-1]) # 行索引
                indice2 = (hype_pos % scores.shape[-1]).long().reshape(-1, 1) # 列索引
                
                for index, each_out in zip(indice1, indice2):
                    index = index.item()
                    each_out = each_out.item()
                    
                    if each_out in repeat_word:
                        pass 
                        # repeat_word[index].append(each_out)
                        # hype_score[index] -= 2 * repeat_word[index].count(each_out)
                    else :
                        repeat_word.append(each_out)
                    
                    if start < beam_size and each_out == douhao_id and len(last_chars) != 0:
                        start += 1
                        word = ix2word[last_chars[index].item()]# 找到上一个字符 记住其押韵情况
                        for i, each_yayun in enumerate(yayun_list):
                            if word in each_yayun:
                                yayun_chars[index] = i
                                break

                # 更新得分
                output_scores = hype_score

                last_chars = indice2

                output_ids = torch.cat([output_ids[indice1], indice2], dim=1).long()
                new_input_ids = torch.cat([token_ids, output_ids], dim=1)
                new_token_type_ids = torch.cat([token_type_ids, torch.ones_like(output_ids)], dim=1)

                end_counts = (output_ids == sep_id).sum(1)  # 统计出现的end标记
                best_one = output_scores.argmax()
                if end_counts[best_one] == 1:
                    # 说明出现终止了～
                    # print(repeat_word)
                    # print(yayun_chars)
                    return output_ids[best_one]
                else :
                    # 保留未完成部分
                    flag = (end_counts < 1)  # 标记未完成序列
                    if not flag.all():  # 如果有已完成的
                        token_ids = token_ids[flag]
                        token_type_ids = token_type_ids[flag]
                        last_chars = last_chars[flag]
                        yayun_chars = yayun_chars[flag]
                        new_input_ids = new_input_ids[flag]
                        new_token_type_ids = new_token_type_ids[flag]
                        output_ids = output_ids[flag]  # 扔掉已完成序列
                        output_scores = output_scores[flag]  # 扔掉已完成序列
                        end_counts = end_counts[flag]  # 扔掉已完成end计数
                        beam_size = flag.sum()  # topk相应变化
                        flag = flag.long()


            return output_ids[output_scores.argmax()]

    def multiTask_batch_generate(self,texts,out_max_length=40,max_length=256,task_prefix = "[_giga]", device="cpu"):

        self.out_max_length = out_max_length
        input_max_length = max_length - out_max_length
        batch_size = len(texts)
        for i in range(len(texts)):
            texts[i] = task_prefix + texts[i]
        batch_tokens_and_types = self.tokenizer.batch_encode_plus(texts,max_length=input_max_length)
        task_prefix_tokens = self.tokenizer(task_prefix, add_special_tokens=False)["input_ids"]
        init_len = [len(sentence_tokens) for sentence_tokens  in texts]
        def add_tokens_to_batch_idx(batch_tokens_and_types:dict, new_tokens: List[List]):
            batch_idx = batch_tokens_and_types['input_ids']
            batch_type = batch_tokens_and_types['token_type_ids']
            assert len(batch_idx)==len(new_tokens)
            for i in range(len(batch_idx)):
                batch_idx[i] = batch_idx[i] + new_tokens[i]
                batch_type[i] = batch_type[i]+[1 for _ in range(len(new_tokens[i]))]

        def idx_to_tensor(batch_dict):
            """
            动态padding， batch为一部分sample
            """

            batch_idx = batch_tokens_and_types['input_ids']
            batch_type = batch_tokens_and_types['token_type_ids']


            def padding(indice, max_length, pad_idx=self.tokenizer.pad_token_id):
                """
                pad 函数
                """
                pad_indice = [item + [pad_idx] * max(0, max_length - len(item)) for item in indice]
                return torch.tensor(pad_indice)

            max_length = max([len(t) for t in batch_idx])
            token_ids_padded = padding(batch_idx, max_length)
            token_type_ids_padded = padding(batch_type, max_length)
            # target_ids_padded = token_ids_padded[:, 1:].contiguous()
            lengths = [len(type_idx) for type_idx in batch_idx]
            return token_ids_padded, token_type_ids_padded, lengths

        add_tokens_to_batch_idx(batch_tokens_and_types, [task_prefix_tokens for _ in range(len(texts))])#all task
        flag = [0]*batch_size
        token_ids_padded, token_type_ids_padded, lengths = idx_to_tensor(batch_tokens_and_types)
        with torch.no_grad():
            for step in range(self.out_max_length):
                token_ids_padded = token_ids_padded.to(device)
                token_type_ids_padded = token_type_ids_padded.to(device)
                scores = self.forward(token_ids_padded,token_type_ids_padded)
                # score_new_tokens_per_line = scores[:,[i-1 for i in lengths],:]
                # new_tokens_id = score_new_tokens_per_line.argmax(-1)#[b,1,1]
                new_tokens = [[scores[i,lengths[i]-1,:].argmax(-1).item()] for i in range(batch_size)]
                add_tokens_to_batch_idx(batch_tokens_and_types,new_tokens=new_tokens)
                token_ids_padded, token_type_ids_padded, lengths = idx_to_tensor(batch_tokens_and_types)
                for i in range(batch_size):
                    if new_tokens[i]==self.tokenizer.sep_token_id:
                        flag[i]=1
                if sum(flag)==batch_size:
                    break
            for i in range(batch_size):
                type_id = batch_tokens_and_types['token_type_ids'][i]
                # tokens_id = batch_tokens_and_types['input_ids'][i]
                left_num = len(type_id) - sum(type_id)
                batch_tokens_and_types['input_ids'][i] = batch_tokens_and_types['input_ids'][i][left_num:]
            resulttext = self.tokenizer.batch_decode(batch_tokens_and_types['input_ids'])
            return resulttext
























        
       
