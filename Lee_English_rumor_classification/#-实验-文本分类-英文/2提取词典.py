import os
import pandas as pd
from tqdm import tqdm
import pickle as pkl


# ## 进度条初始化
tqdm.pandas()

MAX_VOCAB_SIZE = 10000  # 词表长度限制
UNK, PAD, CLS = '<UNK>','[PAD]', '[CLS]'  # 未知字 padding符号, bert中综合信息符号

def build_vocab(file_path, max_size, min_freq):
    df = pd.read_csv(file_path, encoding='utf-8', sep=',')
    # 转化为小写
    sentences = df['content'].apply(lambda x: x.lower())
    # 去除特殊字符
    punct = "/-'?!.,#$%\'()*+-/:;<=>@[\\]^_`{|}~" + '""“”’' + '∞θ÷α•à−β∅³π‘₹´°£€\×™√²—–&'
    punct_mapping = {"‘": "'", "₹": "e", "´": "'", "°": "", "€": "e", "™": "tm", "√": " sqrt ", "×": "x", "²": "2",
                     "—": "-", "–": "-", "’": "'", "_": "-", "`": "'", '“': '"', '”': '"', '“': '"', "£": "e",
                     '∞': 'infinity', 'θ': 'theta', '÷': '/', 'α': 'alpha', '•': '.', 'à': 'a', '−': '-', 'β': 'beta',
                     '∅': '', '³': '3', 'π': 'pi', }
    sentences = sentences.apply(lambda x: clean_special_chars(x, punct, punct_mapping))

    #去除缩写
    contraction_mapping = {"here's":"here is","it's":"it is","ain't": "is not", "aren't": "are not","can't": "cannot", "'cause": "because", "could've": "could have", "couldn't": "could not", "didn't": "did not",  "doesn't": "does not", "don't": "do not", "hadn't": "had not", "hasn't": "has not", "haven't": "have not", "he'd": "he would","he'll": "he will", "he's": "he is", "how'd": "how did", "how'd'y": "how do you", "how'll": "how will", "how's": "how is",  "I'd": "I would", "I'd've": "I would have", "I'll": "I will", "I'll've": "I will have","I'm": "I am", "I've": "I have", "i'd": "i would", "i'd've": "i would have", "i'll": "i will",  "i'll've": "i will have","i'm": "i am", "i've": "i have", "isn't": "is not", "it'd": "it would", "it'd've": "it would have", "it'll": "it will", "it'll've": "it will have","it's": "it is", "let's": "let us", "ma'am": "madam", "mayn't": "may not", "might've": "might have","mightn't": "might not","mightn't've": "might not have", "must've": "must have", "mustn't": "must not", "mustn't've": "must not have", "needn't": "need not", "needn't've": "need not have","o'clock": "of the clock", "oughtn't": "ought not", "oughtn't've": "ought not have", "shan't": "shall not", "sha'n't": "shall not", "shan't've": "shall not have", "she'd": "she would", "she'd've": "she would have", "she'll": "she will", "she'll've": "she will have", "she's": "she is", "should've": "should have", "shouldn't": "should not", "shouldn't've": "should not have", "so've": "so have","so's": "so as", "this's": "this is","that'd": "that would", "that'd've": "that would have", "that's": "that is", "there'd": "there would", "there'd've": "there would have", "there's": "there is", "here's": "here is","they'd": "they would", "they'd've": "they would have", "they'll": "they will", "they'll've": "they will have", "they're": "they are", "they've": "they have", "to've": "to have", "wasn't": "was not", "we'd": "we would", "we'd've": "we would have", "we'll": "we will", "we'll've": "we will have", "we're": "we are", "we've": "we have", "weren't": "were not", "what'll": "what will", "what'll've": "what will have", "what're": "what are",  "what's": "what is", "what've": "what have", "when's": "when is", "when've": "when have", "where'd": "where did", "where's": "where is", "where've": "where have", "who'll": "who will", "who'll've": "who will have", "who's": "who is", "who've": "who have", "why's": "why is", "why've": "why have", "will've": "will have", "won't": "will not", "won't've": "will not have", "would've": "would have", "wouldn't": "would not", "wouldn't've": "would not have", "y'all": "you all", "y'all'd": "you all would","y'all'd've": "you all would have","y'all're": "you all are","y'all've": "you all have","you'd": "you would", "you'd've": "you would have", "you'll": "you will", "you'll've": "you will have", "you're": "you are", "you've": "you have" }
    sentences = sentences.apply(lambda x: clean_contractions(x, contraction_mapping))

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
    vocab_dic.update({UNK: len(vocab_dic), PAD: len(vocab_dic) + 1, CLS: len(vocab_dic) + 1})
    return vocab_dic


def build_dataset(config):
    # if os.path.exists(config.vocab_path):
    #     vocab = pkl.load(open(config.vocab_path, 'rb'))
    # else:
    vocab = build_vocab(config.train_path, max_size=MAX_VOCAB_SIZE, min_freq=1)
    pkl.dump(vocab, open(config.vocab_path, 'wb'))
    print(f"词典======== {vocab}")


# ## 去除特殊字符
def clean_special_chars(text, punct, mapping):
    for p in mapping:
        text = text.replace(p, mapping[p])
    for p in punct:
        text = text.replace(p, f' {p} ')
    specials = {'\u200b': ' ', '…': ' ... ', '\ufeff': '', 'करना': '', 'है': ''}  # Other special characters that I have to deal with in last
    for s in specials:
        text = text.replace(s, specials[s])
    return text


# 去除英语缩写
def clean_contractions(text, mapping):
    specials = ["’", "‘", "´", "`"]
    for s in specials:
        text = text.replace(s, "'")
    text = ' '.join([mapping[t] if t in mapping else t for t in text.split(" ")])
    return text


if __name__ == "__main__":
    class Config():
        def __init__(self):
            self.vocab_path = '../@_数据集/archive/data/vocab.pkl'
            self.train_path = '../@_数据集/archive/data/train.csv'
            self.dev_path = '../@_数据集/archive/data/dev.csv'
            self.test_path = '../@_数据集/archive/data/test.csv'
            self.pad_size = 14

    build_dataset(Config())