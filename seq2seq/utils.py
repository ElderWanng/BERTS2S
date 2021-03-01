import torch
from seq2seq.seq2seq_model import Seq2SeqModel
from seq2seq.bert_cls_classifier import BertClsClassifier
from seq2seq.bert_seq_labeling import BertSeqLabeling
from seq2seq.bert_seq_labeling_crf import BertSeqLabelingCRF
from seq2seq.bert_relation_extraction import BertRelationExtrac

def load_bert(tokenizer, model_name="roberta", model_class="seq2seq", target_size=0):
    """
    model_path: 模型位置
    这是个统一的接口，用来加载模型的
    model_class : seq2seq or encoder
    """
    if model_class == "seq2seq":
        bert_model = Seq2SeqModel(tokenizer, model_name=model_name)
        return bert_model
    elif model_class == "cls":
        if target_size == 0:
            raise Exception("必须传入参数 target_size，才能确定预测多少分类")
        bert_model = BertClsClassifier(tokenizer, target_size, model_name=model_name)
        return bert_model
    elif model_class == "sequence_labeling":
        ## 序列标注模型
        if target_size == 0:
            raise Exception("必须传入参数 target_size，才能确定预测多少分类")
        bert_model = BertSeqLabeling(tokenizer, target_size, model_name=model_name)
        return bert_model
    elif model_class == "sequence_labeling_crf":
        # 带有crf层的序列标注模型
        if target_size == 0:
            raise Exception("必须传入参数 target_size，才能确定预测多少分类")
        bert_model = BertSeqLabelingCRF(tokenizer, target_size, model_name=model_name)
        return bert_model
    elif model_class == "relation_extrac":
        if target_size == 0:
            raise Exception("必须传入参数 target_size 表示预测predicate的种类")
        bert_model = BertRelationExtrac(tokenizer, target_size, model_name=model_name)
        return bert_model
    else :
        raise Exception("model_name_err")


