# coding: UTF-8
import os
import torch
import numpy as np
import pickle as pkl
from tqdm import tqdm
import time
from datetime import timedelta
import pandas as pd
from tool import clean_special_chars, clean_contractions

# ## 进度条初始化
tqdm.pandas()

MAX_VOCAB_SIZE = 10000  # 词表长度限制
UNK, PAD, CLS = '[UNK]','[PAD]', '[CLS]'  # 未知字 padding符号, bert中综合信息符号



def build_vocab(file_path, max_size, min_freq):
    df = pd.read_csv(file_path, encoding='utf-8', sep=';')
    # 去除缩写
    # 转化为小写
    sentences = df['content'].apply(lambda x: x.lower())
    contraction_mapping = {"here's": "here is", "it's": "it is", "ain't": "is not", "aren't": "are not",
                           "can't": "cannot", "'cause": "because", "could've": "could have", "couldn't": "could not",
                           "didn't": "did not", "doesn't": "does not", "don't": "do not", "hadn't": "had not",
                           "hasn't": "has not", "haven't": "have not", "he'd": "he would", "he'll": "he will",
                           "he's": "he is", "how'd": "how did", "how'd'y": "how do you", "how'll": "how will",
                           "how's": "how is", "I'd": "I would", "I'd've": "I would have", "I'll": "I will",
                           "I'll've": "I will have", "I'm": "I am", "I've": "I have", "i'd": "i would",
                           "i'd've": "i would have", "i'll": "i will", "i'll've": "i will have", "i'm": "i am",
                           "i've": "i have", "isn't": "is not", "it'd": "it would", "it'd've": "it would have",
                           "it'll": "it will", "it'll've": "it will have", "it's": "it is", "let's": "let us",
                           "ma'am": "madam", "mayn't": "may not", "might've": "might have", "mightn't": "might not",
                           "mightn't've": "might not have", "must've": "must have", "mustn't": "must not",
                           "mustn't've": "must not have", "needn't": "need not", "needn't've": "need not have",
                           "o'clock": "of the clock", "oughtn't": "ought not", "oughtn't've": "ought not have",
                           "shan't": "shall not", "sha'n't": "shall not", "shan't've": "shall not have",
                           "she'd": "she would", "she'd've": "she would have", "she'll": "she will",
                           "she'll've": "she will have", "she's": "she is", "should've": "should have",
                           "shouldn't": "should not", "shouldn't've": "should not have", "so've": "so have",
                           "so's": "so as", "this's": "this is", "that'd": "that would", "that'd've": "that would have",
                           "that's": "that is", "there'd": "there would", "there'd've": "there would have",
                           "there's": "there is", "here's": "here is", "they'd": "they would",
                           "they'd've": "they would have", "they'll": "they will", "they'll've": "they will have",
                           "they're": "they are", "they've": "they have", "to've": "to have", "wasn't": "was not",
                           "we'd": "we would", "we'd've": "we would have", "we'll": "we will",
                           "we'll've": "we will have", "we're": "we are", "we've": "we have", "weren't": "were not",
                           "what'll": "what will", "what'll've": "what will have", "what're": "what are",
                           "what's": "what is", "what've": "what have", "when's": "when is", "when've": "when have",
                           "where'd": "where did", "where's": "where is", "where've": "where have",
                           "who'll": "who will", "who'll've": "who will have", "who's": "who is", "who've": "who have",
                           "why's": "why is", "why've": "why have", "will've": "will have", "won't": "will not",
                           "won't've": "will not have", "would've": "would have", "wouldn't": "would not",
                           "wouldn't've": "would not have", "y'all": "you all", "y'all'd": "you all would",
                           "y'all'd've": "you all would have", "y'all're": "you all are", "y'all've": "you all have",
                           "you'd": "you would", "you'd've": "you would have", "you'll": "you will",
                           "you'll've": "you will have", "you're": "you are", "you've": "you have"}
    sentences = sentences.apply(lambda x: clean_contractions(x, contraction_mapping))
    # 去除特殊字符
    punct = "/-'?!.,#$%\'()*+-/:;<=>@[\\]^_`{|}~" + '""“”’' + '∞θ÷α•à−β∅³π‘₹´°£€\×™√²—–&'
    punct_mapping = {"‘": "'", "₹": "e", "´": "'", "°": "", "€": "e", "™": "tm", "√": " sqrt ", "×": "x", "²": "2",
                     "—": "-", "–": "-", "’": "'", "_": "-", "`": "'", '“': '"', '”': '"', '“': '"', "£": "e",
                     '∞': 'infinity', 'θ': 'theta', '÷': '/', 'α': 'alpha', '•': '.', 'à': 'a', '−': '-', 'β': 'beta',
                     '∅': '', '³': '3', 'π': 'pi', }
    sentences = sentences.apply(lambda x: clean_special_chars(x, punct, punct_mapping))
    # 提取数组
    sentences = sentences.progress_apply(lambda x: x.split()).values
    vocab_dic = {}
    for sentence in tqdm(sentences, disable=False):
        for word in sentence:
            try:
                vocab_dic[word] += 1
            except KeyError:
                vocab_dic[word] = 1
    vocab_list = sorted([_ for _ in vocab_dic.items() if _[1] >= min_freq], key=lambda x: x[1], reverse=True)[:max_size]
    vocab_dic = {word_count[0]: idx for idx, word_count in enumerate(vocab_list)}
    vocab_dic.update({UNK: len(vocab_dic), PAD: len(vocab_dic) + 1})
    return vocab_dic


def build_dataset(config):
    if os.path.exists(config.vocab_path):
        vocab = pkl.load(open(config.vocab_path, 'rb'))
    else:
        vocab = build_vocab(config.train_path, max_size=MAX_VOCAB_SIZE, min_freq=1)
        pkl.dump(vocab, open(config.vocab_path, 'wb'))
    #print(f"词典大小======== {len(vocab)}")

    def load_dataset(path, pad_size=32):
        df = pd.read_csv(path, encoding='utf-8', sep=',')
        # TODO 这里读数据集写死了 title
        # 转化为小写
        sentences = df['content'].apply(lambda x: x.lower())
        # 去除缩写
        contraction_mapping = {"here's": "here is", "it's": "it is", "ain't": "is not", "aren't": "are not",
                               "can't": "cannot", "'cause": "because", "could've": "could have",
                               "couldn't": "could not",
                               "didn't": "did not", "doesn't": "does not", "don't": "do not", "hadn't": "had not",
                               "hasn't": "has not", "haven't": "have not", "he'd": "he would", "he'll": "he will",
                               "he's": "he is", "how'd": "how did", "how'd'y": "how do you", "how'll": "how will",
                               "how's": "how is", "I'd": "I would", "I'd've": "I would have", "I'll": "I will",
                               "I'll've": "I will have", "I'm": "I am", "I've": "I have", "i'd": "i would",
                               "i'd've": "i would have", "i'll": "i will", "i'll've": "i will have", "i'm": "i am",
                               "i've": "i have", "isn't": "is not", "it'd": "it would", "it'd've": "it would have",
                               "it'll": "it will", "it'll've": "it will have", "it's": "it is", "let's": "let us",
                               "ma'am": "madam", "mayn't": "may not", "might've": "might have", "mightn't": "might not",
                               "mightn't've": "might not have", "must've": "must have", "mustn't": "must not",
                               "mustn't've": "must not have", "needn't": "need not", "needn't've": "need not have",
                               "o'clock": "of the clock", "oughtn't": "ought not", "oughtn't've": "ought not have",
                               "shan't": "shall not", "sha'n't": "shall not", "shan't've": "shall not have",
                               "she'd": "she would", "she'd've": "she would have", "she'll": "she will",
                               "she'll've": "she will have", "she's": "she is", "should've": "should have",
                               "shouldn't": "should not", "shouldn't've": "should not have", "so've": "so have",
                               "so's": "so as", "this's": "this is", "that'd": "that would",
                               "that'd've": "that would have",
                               "that's": "that is", "there'd": "there would", "there'd've": "there would have",
                               "there's": "there is", "here's": "here is", "they'd": "they would",
                               "they'd've": "they would have", "they'll": "they will", "they'll've": "they will have",
                               "they're": "they are", "they've": "they have", "to've": "to have", "wasn't": "was not",
                               "we'd": "we would", "we'd've": "we would have", "we'll": "we will",
                               "we'll've": "we will have", "we're": "we are", "we've": "we have", "weren't": "were not",
                               "what'll": "what will", "what'll've": "what will have", "what're": "what are",
                               "what's": "what is", "what've": "what have", "when's": "when is", "when've": "when have",
                               "where'd": "where did", "where's": "where is", "where've": "where have",
                               "who'll": "who will", "who'll've": "who will have", "who's": "who is",
                               "who've": "who have",
                               "why's": "why is", "why've": "why have", "will've": "will have", "won't": "will not",
                               "won't've": "will not have", "would've": "would have", "wouldn't": "would not",
                               "wouldn't've": "would not have", "y'all": "you all", "y'all'd": "you all would",
                               "y'all'd've": "you all would have", "y'all're": "you all are",
                               "y'all've": "you all have",
                               "you'd": "you would", "you'd've": "you would have", "you'll": "you will",
                               "you'll've": "you will have", "you're": "you are", "you've": "you have"}
        sentences = sentences.apply(lambda x: clean_contractions(x, contraction_mapping))
        # 去除特殊字符
        punct = "/-'?!.,#$%\'()*+-/:;<=>@[\\]^_`{|}~" + '""“”’' + '∞θ÷α•à−β∅³π‘₹´°£€\×™√²—–&'
        punct_mapping = {"‘": "'", "₹": "e", "´": "'", "°": "", "€": "e", "™": "tm", "√": " sqrt ", "×": "x", "²": "2",
                         "—": "-", "–": "-", "’": "'", "_": "-", "`": "'", '“': '"', '”': '"', '“': '"', "£": "e",
                         '∞': 'infinity', 'θ': 'theta', '÷': '/', 'α': 'alpha', '•': '.', 'à': 'a', '−': '-',
                         'β': 'beta',
                         '∅': '', '³': '3', 'π': 'pi', }
        sentences = sentences.apply(lambda x: clean_special_chars(x, punct, punct_mapping))
        # 提取数组
        sentences = sentences.progress_apply(lambda x: x).values
        labels = df['label']
        labels_id = list(set(df['label']))
        labels_id.sort()
        contents = []
        count = 0
        for i, token in tqdm(enumerate(sentences)):
            label = labels[i]


            #print(token)
            token = config.tokenizer.tokenize(token)
            token = [CLS] + token
            mask=[]

            seq_len = len(token)
            count += seq_len

            token_ids = config.tokenizer.convert_tokens_to_ids(token)

            if pad_size:
                if len(token) < pad_size:
                    mask = [1] * len(token_ids) + [0] * (pad_size - len(token))
                    token_ids += ([0] * (pad_size - len(token)))
                else:
                    mask = [1] * pad_size
                    token_ids = token_ids[:pad_size]
                    seq_len = pad_size

            contents.append((token_ids, labels_id.index(label), seq_len, mask))
        print(f"数据集地址========{path}")
        print(f"数据集总词数========{count}")
        print(f"数据集文本数========{len(sentences)}")
        print(f"数据集文本平均词数========{count/len(sentences)}")
        print(f"训练集标签========{set(df['label'])}")
        return contents  # [([...], 0), ([...], 1), ...]
    train = load_dataset(config.train_path, config.pad_size)
    dev = load_dataset(config.dev_path, config.pad_size)
    test = load_dataset(config.test_path, config.pad_size)

    return vocab, train, dev, test


class DatasetIterater(object):
    def __init__(self, batches, batch_size, device):
        self.batch_size = batch_size
        self.batches = batches
        self.n_batches = len(batches) // batch_size
        self.residue = False  # 记录batch数量是否为整数
        if len(batches) % self.n_batches != 0:
            self.residue = True
        self.index = 0
        self.device = device

    def _to_tensor(self, datas):
        x = torch.LongTensor([_[0] for _ in datas]).to(self.device)
        y = torch.LongTensor([_[1] for _ in datas]).to(self.device)

        # pad前的长度(超过pad_size的设为pad_size)
        seq_len = torch.LongTensor([_[2] for _ in datas]).to(self.device)

        mask = torch.LongTensor([_[3] for _ in datas]).to(self.device)
        return (x, seq_len, mask), y

    def __next__(self):
        if self.residue and self.index == self.n_batches:
            batches = self.batches[self.index * self.batch_size: len(self.batches)]
            self.index += 1
            batches = self._to_tensor(batches)
            return batches

        elif self.index >= self.n_batches:
            self.index = 0
            raise StopIteration
        else:
            batches = self.batches[self.index * self.batch_size: (self.index + 1) * self.batch_size]
            self.index += 1
            batches = self._to_tensor(batches)
            return batches

    def __iter__(self):
        return self

    def __len__(self):
        if self.residue:
            return self.n_batches + 1
        else:
            return self.n_batches


def build_iterator(dataset, config):
    iter = DatasetIterater(dataset, config.batch_size, config.device)
    return iter


def get_time_dif(start_time):
    """获取已使用时间"""
    end_time = time.time()
    time_dif = end_time - start_time
    return timedelta(seconds=int(round(time_dif)))


if __name__ == "__main__":
    class Config():
        def __init__(self):
            self.vocab_path = '../@_数据集/TLND/data/vocab.pkl'
            self.train_path = '../@_数据集/TLND/data/train.csv'
            self.dev_path = '../@_数据集/TLND/data/dev.csv'
            self.test_path = '../@_数据集/TLND/data/test.csv'
            self.pad_size = 14

    # build_dataset(Config())
    # print(np.array(np.load('../#-EXAMPLE-NN/THUCNews/data/embedding_SougouNews.npz')['embeddings'][4758:]))