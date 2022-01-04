# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
from sklearn.svm import SVR
from sklearn.linear_model import LogisticRegression, Lasso
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
import xlrd
import matplotlib.pyplot as plt
import os
from sklearn import preprocessing

'''
读取excel文件成为dataframe刘
'''
def read_excel(filename, col_names = ('NO_', 'COURE_NAME', 'ZHSCORE')):
    book = xlrd.open_workbook(filename, 'r')
    sheet = book.sheet_by_index(0)
    result = dict()
    for i in range(sheet.ncols):
        if sheet.cell_value(0,i) in col_names:
            result[sheet.cell_value(0,i)] = sheet.col_values(i,1)
    result = pd.DataFrame(result)
    result = result.replace('', np.nan)
    result = result.dropna()
    try:
        result = result.pivot_table(index=col_names[0], columns=col_names[1], values=col_names[2], fill_value=0)
    except Exception:
        print(filename + '文件内容为空')
    t = result[result > 0].count()
    idx = t[t < len(result) / 2].index
    result = result.drop(idx, 1)
    result[result > 100] = result - 60
    return result


def get_table(pre_grade, follow_grade):
    s1 = set(pre_grade.columns)
    s2 = set(follow_grade.columns)
    x_col = list(s1 & s2)
    y_col = list(s1 - (s1 & s2))
    train, test = train_test_split(pre_grade, test_size=0.1,
                                   random_state=3
                                   )
    x_train = train[x_col]
    y_train = train[y_col]
    x_test = test[x_col]
    y_test = test[y_col]
    x_pred = follow_grade[x_col]
    return x_train, y_train, x_test, y_test, x_pred

def preScale(x_train,x_test,x_pred):
    scaler = preprocessing.StandardScaler().fit(x_train)
    x_col = x_train.columns
    x_index = x_train.index
    x_train = scaler.transform(x_train)
    x_train = pd.DataFrame(x_train,index=x_index,columns=x_col)
    x_index = x_test.index
    x_test = scaler.transform(x_test)
    x_test = pd.DataFrame(x_test, index=x_index, columns=x_col)
    x_index = x_pred.index
    x_pred = scaler.transform(x_pred)
    x_pred = pd.DataFrame(x_pred, index=x_index, columns=x_col)
    return x_train,x_test,x_pred

def PCAdeflat(dataFrame,demension = 1):
    pca = PCA(n_components=demension)  # 实例化
    pca = pca.fit(dataFrame)  # 拟合模型
    res = pca.transform(dataFrame)
    return res