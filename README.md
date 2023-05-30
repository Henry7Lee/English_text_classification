# English_text_classification
基于pytorch的英文文本分类模型


## 使用模型
TextCNN, TextRNN, FastText, TextRCNN, TextRNN_Att, DPCNN, Transformer

bert, ERNIE, bert_CNN, bert_DPCNN, bert_RCNN, bert_RNN

## 数据集
kdnuggets-fake-news：(2分类)

https://github.com/lutzhamel/fake-news/tree/master


*或者使用自己的数据集，详情见文件和参考资料*

## 预训练词向量
glove： https://nlp.stanford.edu/projects/glove/

fasttext： https://fasttext.cc/


## 实验结果
|  模型   | 准确率  | 词向量  |
|  ----  | ----  |   ----  |
| TextRNN  | 87.25% | glove.6B.300d |
| TextRNN_att  | 92.10% | glove.6B.300d |
| TextRCNN | 92.31% | glove.6B.300d |
| FastText  | 88.72% | glove.6B.300d |
| TextDPCNN  | 89.46% | glove.6B.300d |
| TextCNN  | 90.41% | glove.6B.300d |
| Transformer  | 91.04% | glove.6B.300d |
| Bert  | 93.89% | Bert-based-uncased-EN |
| Bert_RNN  |  92.62% | Bert-based-uncased-EN |
| Bert_CNN  |  93.89% | Bert-based-uncased-EN |
| Bert_RCNN  | **95.26%** | Bert-based-uncased-EN |
| Bert_DPCNN  | 93.47% | Bert-based-uncased-EN |
| ERNIE  | 90.09% |ernie-based-2.0-EN |




## 主要参考资料
https://github.com/649453932/Chinese-Text-Classification-Pytorch#chinese-text-classification-pytorch

https://github.com/649453932/Bert-Chinese-Text-Classification-Pytorch

https://zhuanlan.zhihu.com/p/73176084

https://blog.csdn.net/qq_43592352/article/details/122960985
