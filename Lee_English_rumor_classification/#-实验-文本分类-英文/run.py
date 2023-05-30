# coding: UTF-8
import time
import torch
import numpy as np
from train_eval import train, init_network
from importlib import import_module
import argparse
from utils import build_dataset, build_iterator, get_time_dif
import random

parser = argparse.ArgumentParser(description='English Text Classification')
parser.add_argument("--model", type=str, default="TextCNN", help="choose a model: bert, ERNIE")
parser.add_argument('--embedding', default='random', type=str, help='random or pre_trained')
args = parser.parse_args()

# model:
# TextCNN, TextRNN, FastText, TextRCNN, TextRNN_Att, DPCNN, Transformer
# bert, ERNIE, bert_CNN, bert_DPCNN, bert_RCNN, bert_RNN

if __name__ == '__main__':
    dataset = '../@_数据集/archive'  # 数据集

    # glove
    # embedding = 'glove.6B.300d.npz'
    # embedding = 'glove.6B.50d.npz'

    # 腾讯
    embedding = 'wiki-news-300d-1M.npz'

    # 随机初始化:random
    if args.embedding == 'random':
        embedding = 'random'

    model_name = args.model
    if model_name == 'FastText':
        from utils_fasttext import build_dataset, build_iterator, get_time_dif
        embedding = 'random'
    elif 'bert' in model_name or 'ERNIE' in model_name :
        from utils_bert import build_dataset, build_iterator, get_time_dif
    else:
        from utils import build_dataset, build_iterator, get_time_dif

    x = import_module('models.' + model_name)
    config = x.Config(dataset, embedding)
    random.seed(1)
    # 设置numpy的随机种子，以使得结果是确定的
    np.random.seed(1)
    # 为CPU设置种子用于生成随机数，以使得结果是确定的
    torch.manual_seed(1)
    # 为当前GPU设置随机种子，以使得结果是确定的
    torch.cuda.manual_seed_all(1)
    torch.backends.cudnn.deterministic = True  # 保证每次结果一样

    start_time = time.time()
    print("Loading data...")
    vocab, train_data, dev_data, test_data = build_dataset(config)
    train_iter = build_iterator(train_data, config)
    dev_iter = build_iterator(dev_data, config)
    test_iter = build_iterator(test_data, config)
    time_dif = get_time_dif(start_time)
    print("Time usage:", time_dif)

    # train
    config.n_vocab = len(vocab)
    model = x.Model(config).to(config.device)
    if model_name == 'Transformer' or 'bert' in model_name or 'ERNIE' in model_name :
        pass
    else:
        init_network(model)
    #print(model.parameters)
    train(config, model, train_iter, dev_iter, test_iter)