# -*- coding: utf-8 -*-
"""
@author: 
@time: 
"""
from pandas import DataFrame

from process import *
from flask import Flask, render_template, jsonify
from pyecharts import options as opts
from pyecharts.charts import Line, Bar, Pie, Scatter
from pyecharts.globals import CurrentConfig
CurrentConfig.ONLINE_HOST = "http://127.0.0.1:8000/assets/"
pre_grade = read_excel('train.xls')
follow_grade = read_excel('test.xls')
try:
    x_train, y_train, x_test, y_test, x_pred = get_table(pre_grade, follow_grade)
    x_train, x_test, x_pred = preScale(x_train, x_test, x_pred)
    scores = {}
    y_res = {}
    y_res['svr'], scores['svr'] = predictSVR(x_train, y_train, x_test, y_test, x_pred)
    y_res['lR'], scores['lR'] = predictLR(x_train, y_train, x_test, y_test, x_pred)
    y_res['adaBoost'], scores['adaBoost'] = predictAdaBoost(x_train, y_train, x_test, y_test, x_pred)
    y_res['Lasso'], scores['Lasso'] = predictLasso(x_train, y_train, x_test, y_test, x_pred)
    y_res['LGBM'], scores['LGBM'] = predictLGBM(x_train, y_train, x_test, y_test, x_pred)
    maxres = max(scores, key=scores.get)
    y_prediction = y_res[maxres]
    draw_line(y_prediction,'pred')
    draw_line(y_train,'real')
    scores = DataFrame.from_dict(scores,orient='index', columns=['F1均分'])
    draw_diff(scores,'mean')
    print(y_prediction)
    # y_prediction.to_csv('result.xlsx', encoding='utf_8_sig')
except Exception as e:
    print("出现错误： ",e)

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

