# coding=gbk
"""
2.数据集合处理
拆分出需要预测科目与参照科目
"""
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from process import *

train = read_excel('train.xls')
pred = read_excel('test.xls')
train,test = train_test_split(train, test_size=0.25)
s1 = set(train.columns)
s2 = set(pred.columns)
x_col = list(s1 & s2)
y_col = list(s1 - (s1 & s2))
x_train = train[x_col]
y_train = train[y_col]
x_test = test[x_col]
y_test = test[y_col]
x_pred = pred[x_col]

