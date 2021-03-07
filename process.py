# coding=gbk
import numpy as np
import pandas as pd
from pandas import DataFrame
from predict import *
from sklearn.svm import SVR
import lightgbm as lgbm
from sklearn.linear_model import LogisticRegression, Lasso
from sklearn.ensemble import AdaBoostClassifier
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from pyecharts import options as opts
from pyecharts.charts import Line, Bar, Pie, Scatter
import xlrd
import os

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
    train, test = train_test_split(pre_grade, test_size=0.2)
    x_train = train[x_col]
    y_train = train[y_col]
    x_test = test[x_col]
    y_test = test[y_col]
    x_pred = follow_grade[x_col]
    return x_train, y_train, x_test, y_test, x_pred

def predict(x_train, y_train, x_test ,y_test, x_pred):
    # agorList = [ 'svr', 'lR', 'adaBoost', 'Lasso', 'LGBM']
    scores = {}
    y_res = {}
    y_res['svr'],scores['svr'] = predictSVR(x_train, y_train, x_test, y_test, x_pred)
    y_res['lR'], scores['lR'] = predictLR(x_train, y_train, x_test, y_test, x_pred)
    y_res['adaBoost'], scores['adaBoost'] = predictAdaBoost(x_train, y_train, x_test, y_test, x_pred)
    y_res['Lasso'], scores['Lasso'] = predictLasso(x_train, y_train, x_test, y_test, x_pred)
    y_res['LGBM'], scores['LGBM'] = predictLGBM(x_train, y_train, x_test, y_test, x_pred)
    maxres = max(scores, key=scores.get)
    scores = DataFrame.from_dict(scores, orient='index', columns=['F1均分'])
    draw_diff(scores,'mean')
    return y_res[maxres]

def get_prediction(pre_grade_filename, follow_grade_filename, prediction_filename):
    try:
        pre_grade = read_excel(pre_grade_filename)
        follow_grade = read_excel(follow_grade_filename)
        x_train, y_train, x_test, y_test, x_pred = get_table(pre_grade, follow_grade)
        x_train, x_test, x_pred = preScale(x_train,x_test,x_pred)
        y_prediction = predict(x_train, y_train, x_test, y_test, x_pred)
        draw_line(y_prediction,'pred')
        draw_line(y_train,'train')
        y_prediction.to_csv(prediction_filename, encoding='utf_8_sig')
    except Exception as e:
        print(e)
        print('课程完全重合，无预测科目')

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


def process_grade(grade_dir, result_dir):
    major = os.listdir(grade_dir)
    for cur_major in major:
        cur_dir = grade_dir + '/' + cur_major
        files = os.listdir(cur_dir)
        filepath = []
        for cur_files in files:
            filepath.append(cur_dir + '/' + cur_files)
        print(cur_major + ':')
        if 'train' in filepath[0]:
            get_prediction(filepath[0], filepath[1], result_dir + '/' + cur_major + '预测结果.csv')
        else:
            get_prediction(filepath[1], filepath[0], result_dir + '/' + cur_major + '预测结果.csv')

def draw_line(dataframe,type):
    line = (Line(init_opts=opts.InitOpts(width="2000px", height="1000px"))
            .add_xaxis(dataframe.columns.tolist())
            )
    for index, row in dataframe.iterrows():
        line.add_yaxis(index, row.tolist(), is_connect_nones=True)
    line.set_global_opts(
        title_opts=opts.TitleOpts(title="成绩折线图"),
        # LegendOpts：图例配置项
        legend_opts=opts.LegendOpts(is_show=False),
    )
    res = 'line_data_'+type+'_pic.html'
    line.render(res)

from pyecharts.charts import Graph
def draw_diff(dataframe,type):
    bar =(Bar(init_opts=opts.InitOpts(width="2000px", height="1000px")))
    xaxis = dataframe.columns.tolist()
    bar.add_xaxis(xaxis)
    for col,row in dataframe.iterrows():
        bar.add_yaxis(col,row.values.tolist())
    bar.set_global_opts(
        title_opts=opts.TitleOpts(title="F1条形图"),
        # LegendOpts：图例配置项
        # legend_opts=opts.LegendOpts(is_show=False),
    )
    name = 'bar_with_'+type+'.html'
    bar.render(name)


if __name__ == '__main__':
    gradedir = './res/grade'
    resultdir = './res/result'
    process_grade(gradedir, resultdir)




