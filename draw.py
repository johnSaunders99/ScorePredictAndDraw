# -*- coding: utf-8 -*-
from pandas import DataFrame
from process import *
from flask import Flask, render_template, jsonify
from pyecharts import options as opts
from pyecharts.charts import Line, Bar, Pie, Scatter, EffectScatter
from pyecharts.globals import CurrentConfig, ThemeType

templates = './templates/'
kde = templates + 'KDEcharts/'
linepath = templates + 'linecharts/'
barpath = templates + 'barcharts/'
scatterpath = templates + 'scatter/'

if not os.path.exists(kde):
    os.makedirs(kde)
if not os.path.exists(linepath):
    os.makedirs(linepath)
if not os.path.exists(barpath):
    os.makedirs(barpath)
if not os.path.exists(scatterpath):
    os.makedirs(scatterpath)


def draw_line(dataframe, type):
    line = (Line(init_opts=opts.InitOpts(width="2000px", height="1000px"))
            .add_xaxis(dataframe.columns.tolist())
            )
    for index, row in dataframe.iterrows():
        line.add_yaxis(index, row.tolist(), is_connect_nones=True)
    line.set_global_opts(
        title_opts=opts.TitleOpts(title="成绩折线图"),
        # LegendOpts：图例配置项
        legend_opts=opts.LegendOpts(is_show=False),
        xaxis_opts=opts.AxisOpts(name_rotate=60, name="subject", axislabel_opts={"rotate": 45})
    )
    name = linepath + 'line_data_' + type + '_pic.html'
    line.render(name)
    return line


def draw_bar(dataframe, type, title):
    bar = (Bar(init_opts=opts.InitOpts(width="2000px", height="1000px", theme=ThemeType.LIGHT)))
    xaxis = dataframe.columns.tolist()
    bar.add_xaxis(xaxis)
    if dataframe.values[0] <= 1:
        for col, row in dataframe.iterrows():
            yaixs = [round(n * 100, 2) for n in row.values.tolist()]
            bar.add_yaxis(col, yaixs, label_opts=opts.LabelOpts(formatter="{c} %")
                          # ,label_opts=opts.LabelOpts(formatter="{c} %",)
                          # markline_opts=opts.MarkLineOpts(precision=4),
                          )
    else:
        for col, row in dataframe.iterrows():
            yaixs = row.values.tolist()
            bar.add_yaxis(col, yaixs
                          # ,label_opts=opts.LabelOpts(formatter="{c} %",)
                          # markline_opts=opts.MarkLineOpts(precision=4),
                          )
        # row = map(lambda x: x * 100, row.values.tolist())


    bar.set_global_opts(
        title_opts=opts.TitleOpts(title=title),
        # LegendOpts：图例配置项
        # legend_opts=opts.LegendOpts(is_show=False),
        # yaxis_opts=opts.AxisOpts(axislabel_opts=opts.LabelOpts(formatter="{value} %"), interval=10)
    )
    # bar.set_series_opts(
    #     label_opts=opts.LabelOpts(is_show=False),
    #     markpoint_opts=opts.MarkPointOpts(
    #         data=[
    #             opts.MarkPointItem(type_="max", name="最大值")
    #         ]
    #     ),
    # )
    name = barpath + 'bar_with_' + type + '.html'
    bar.render(name)
    return bar


def draw_twodiff(dataframea, dataframeb, type):
    bar = (Bar(init_opts=opts.InitOpts(width="2000px", height="1000px")))
    xaxis = dataframea.columns.tolist()
    bar.add_xaxis(xaxis)
    train = []
    pred = []
    for col in xaxis:
        train.append(dataframea[col].count())
        if col in dataframeb.columns.tolist():
            pred.append(dataframeb[col].count())
    bar.add_yaxis('train', train)
    bar.add_yaxis('predict', pred)
    bar.set_global_opts(
        title_opts=opts.TitleOpts(title="数据总数对比图"),
        # LegendOpts：图例配置项
        # legend_opts=opts.LegendOpts(is_show=False),
        xaxis_opts=opts.AxisOpts(name_rotate=60, name="subject", axislabel_opts={"rotate": 45}),
    )
    name = barpath + 'twobar_with_' + type + '.html'
    bar.render(name)
    return bar


def draw_scatter(dataframe, type):
    xaxis = dataframe.columns.tolist()
    scatter = (
        EffectScatter()
            .add_xaxis(xaxis)
            .set_global_opts(
            legend_opts=opts.LegendOpts(is_show=False),
            title_opts=opts.TitleOpts(title="分数散点图"),
            xaxis_opts=opts.AxisOpts(splitline_opts=opts.SplitLineOpts(is_show=True)),
            yaxis_opts=opts.AxisOpts(splitline_opts=opts.SplitLineOpts(is_show=True)),
        )
    )
    for col, row in dataframe.iterrows():
        scatter.add_yaxis(col, row.values.tolist(),
                          )
    name = scatterpath + 'scatter_with_' + type + '.html'
    scatter.render(name)
    return scatter


from numpy import linspace, hstack
import seaborn as sns


def draw_KDE(x, y=None, sign='unset'):
    sns.set()
    # size = list(range(x.size))
    if y is not None:
        # x = DataFrame(x, index=size, columns=['val'])
        # y = DataFrame(y, index=size, columns=['val'])
        # sample = hstack([x.T, y.T])
        # x = np.delete(x.T, np.argwhere(x.T == 0))
        # y = np.delete(x.T, np.argwhere(y.T == 0))
        # sample = np.delete(sample, np.argwhere(sample == 0))
        a = np.random.randn(100)
        fig = sns.kdeplot(x.T[0], y.T[0])
        name = 'Bina_' + sign + '.png'
    else:
        # sample = hstack(x.T)
        name = 'Single_' + sign + '.png'
        fig = sns.kdeplot(x.T[0])
    scatter_fig = fig.get_figure()
    scatter_fig.savefig(kde + name, dpi=400)
    return scatter_fig
