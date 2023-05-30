import csv
import os.path
import random

import pandas as pd

df = pd.read_csv("../@_数据集/archive/fake_or_real_news.csv", encoding='utf-8', sep=',')

labels = set(df['label'])
contents = df['text']
count = {}
cal = {}
for p in df['label']:
    cal[p] = 0
    try:
        count[p] += 1
    except KeyError:
        count[p] = 1
print(count) #输出类别个数


#按照70：15：15的比例来均分数据集
train, val, test = [], [], []
for i, label in enumerate(df['label']):
    if cal[label] < count[label] * 0.7:
        train.append({'label': label, 'content': contents[i]})
    elif cal[label] < count[label] * 0.85:
        val.append({'label': label, 'content': contents[i]})
    else:
        test.append({'label': label, 'content': contents[i]})
    cal[label] += 1
random.shuffle(train)
random.shuffle(val)
random.shuffle(test)


def mk_no_exist_dir(path):
    import os
    if not os.path.exists(path):
        os.makedirs(path)
    else:
        pass

mk_no_exist_dir('../@_数据集/archive/data/')
mk_no_exist_dir('../@_数据集/archive/log/')
mk_no_exist_dir('../@_数据集/archive/saved_dict/')


with open('../@_数据集/archive/data/train.csv', 'w', newline='', encoding='utf-8') as f:
    xieru = csv.DictWriter(f, fieldnames=['label','content'],delimiter=',')
    xieru.writeheader()  # 将字段写入csv格式文件首行
    xieru.writerows(train)  # writerows方法是一下子写入多行内容

with open('../@_数据集/archive/data/dev.csv', 'w', newline='', encoding='utf-8') as f:
    xieru = csv.DictWriter(f, ['label','content'],delimiter=',')
    xieru.writeheader()  # 将字段写入csv格式文件首行
    xieru.writerows(val)  # writerows方法是一下子写入多行内容

with open('../@_数据集/archive/data/test.csv', 'w', newline='', encoding='utf-8') as f:
    xieru = csv.DictWriter(f, ['label','content'],delimiter=',')
    xieru.writeheader()  # 将字段写入csv格式文件首行
    xieru.writerows(test)  # writerows方法是一下子写入多行内容


