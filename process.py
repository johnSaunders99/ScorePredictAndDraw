# -*- coding: utf-8 -*-
'''
-------------
主要运行函数，运行后对对应的数据表存放位置进行数据清洗与预测
其中数据表需要至少有一份名称包含train的训练数据、另一份作为待预测数据读取。
todo 待优化的地方：
1.为将flask 制作简易输入界面，可以让数据手动或传入数据文件进行解析并再训练，得出结果。
2. 对数据的读入读出使用可定制的输入界面，也可以通过点击按钮进行文件传输直接下载。 考虑到C/S服务，或者直接内嵌功能至系统
3.对lightGBM的更深一步调优，尽量达到SVR的效果。
4.修改成对同课名成绩提供其高初始权值进行训练(某些课程)
-------------
'''
import numpy as np
import openpyxl
from scipy.stats.kde import gaussian_kde
import pandas as pd
from pandas import DataFrame
from readShuffle import get_table, read_excel, preScale, PCAdeflat
from predict import *
from draw import *
import os

'''
predict 对所有可使用模型进行预测，选择最优的分数所对应的预测值返回进行写入。
'''

ans = {};


def predict(x_train, y_train, x_test, y_test, x_pred, retrain):
    scores = {}
    maes = {}
    rmses = {}
    precision = {}
    accuracy = {}
    y_res = {}
    xflat = PCAdeflat(x_train)
    yfalt = PCAdeflat(y_train)
    # draw_KDE(xflat, yfalt, 'xy')
    # draw_KDE(xflat, sign='x')
    # draw_KDE(yfalt, sign='y')
    # x_test = x_test.sample(frac=1, axis=1)
    # y_test = y_test.sample(frac=1, axis=1)
    y_res['svr'], scores['svr'], maes['svr'], rmses['svr'] = predictSVR(x_train, y_train, x_test, y_test, x_pred,
                                                                        retrain)
    y_res['svc'], scores['svc'], precision['svc'], accuracy['svc'] = predictSVC(x_train, y_train, x_test, y_test,
                                                                                x_pred, retrain)
    y_res['lR'], scores['lR'], precision['lR'], accuracy['lR'] = predictLR(x_train, y_train, x_test, y_test, x_pred,
                                                                           retrain)
    y_res['adaBoostR'], scores['adaBoostR'], maes['adaBoostR'], rmses['adaBoostR'] = predictAdaBoostR(x_train, y_train,
                                                                                                      x_test, y_test,
                                                                                                      x_pred, retrain)
    y_res['adaBoost'], scores['adaBoost'], precision['adaBoost'], accuracy['adaBoost'] = predictAdaBoost(x_train,
                                                                                                         y_train,
                                                                                                         x_test, y_test,
                                                                                                         x_pred,
                                                                                                         retrain)
    y_res['Lasso'], scores['Lasso'], maes['Lasso'], rmses['Lasso'] = predictLasso(x_train, y_train, x_test,
                                                                                          y_test, x_pred, retrain)
    y_res['DT'], scores['DT'], precision['DT'], accuracy['DT'] = predictDT(x_train, y_train, x_test, y_test, x_pred,
                                                                           retrain)
    y_res['NB'], scores['NB'], precision['NB'], accuracy['NB'] = predictNB(x_train, y_train, x_test, y_test, x_pred,
                                                                           retrain)
    y_res['LGBM'], scores['LGBM'], maes['LGBM'], rmses['LGBM'] = predictLGBM(x_train, y_train, x_test, y_test, x_pred,
                                                                             retrain)
    maxres = max(scores, key=scores.get)
    scores = DataFrame.from_dict(scores, orient='index', columns=['F1/R2均分'])
    maes =  DataFrame.from_dict(maes, orient='index', columns=['mean abs error 均分'])
    rmses = DataFrame.from_dict(rmses, orient='index', columns=['模型均分'])
    precision = DataFrame.from_dict(precision, orient='index', columns=['准确度均分'])
    accuracy = DataFrame.from_dict(accuracy, orient='index', columns=['精确度均分'])
    return y_res[maxres], scores, maes, rmses, precision, accuracy


def get_prediction(pre_grade_filename, follow_grade_filename, prediction_filename, retrain):
    picDict = {}
    try:
        pre_grade = read_excel(pre_grade_filename)
        follow_grade = read_excel(follow_grade_filename)
        x_train, y_train, x_test, y_test, x_pred = get_table(pre_grade, follow_grade)
        x_train, x_test, x_pred = preScale(x_train, x_test, x_pred)
        picDict['16'] = draw_line(pre_grade, 'after_scale16')
        picDict['17'] = draw_line(follow_grade, 'after_scale17')
        y_prediction, scores, maes, rmses, precision, accuracy = predict(x_train, y_train, x_test, y_test, x_pred,
                                                                         retrain)
        picDict['line_pred'] = draw_line(y_prediction, 'pred')
        picDict['line_train'] = draw_line(y_train, 'follow')
        picDict['bar_score'] = draw_bar(scores, 'mean', 'R2、AC决定系数对比图')
        picDict['scatter_predict'] = draw_scatter(y_prediction, 'pred')
        picDict['scatter_train'] = draw_scatter(follow_grade, 'follow')
        picDict['bar_mae'] = draw_bar(maes, 'mae', '回归模型平均绝对误差指标对比')
        picDict['bar_rmse'] = draw_bar(rmses, 'rmse', '回归模型均方根误差指标对比')
        picDict['bar_accuracy'] = draw_bar(accuracy, 'accuracy', '分类准确度指标对比')
        picDict['bar_precision'] = draw_bar(precision, 'precision', '分类宏观macro精确度指标对比')
        y_prediction.to_excel(prediction_filename, encoding='utf_8_sig', sheet_name='预测结果')
        return picDict
    except Exception as e:
        print(e)
        print('课程完全重合，无预测科目')


'''
Parameters
        ----------
        grade_dir : str
            where the train_test excel files is.
            
        result_dir : str
            where to save the result excel files.

        retrain : bool
            is retrain or not , if retrain, force reset All the models.

        Returns
        -------
        DataFrame
'''


def process_grade(grade_dir: str, result_dir: str, retrain: bool):
    major = os.listdir(grade_dir)
    for cur_major in major:
        cur_dir = grade_dir + '/' + cur_major
        files = os.listdir(cur_dir)
        filepath = []
        for cur_files in files:
            filepath.append(cur_dir + '/' + cur_files)
        print(cur_major + ':' + '文件夹内开始预测， 是否重新训练：' + str(retrain))
        trainf = ""
        testf = ""
        for file in filepath:
            if 'train' in file:
                trainf = file
            if 'test' in file:
                testf = file
            if trainf != "" and testf != "":
                break
        if trainf == "" or testf == "":
            print("no correct files pattern was find, please make sure one train and one test in file location.")
        else:
            global ans
            ans = get_prediction(trainf, testf, result_dir + '/' + cur_major + '预测结果.xlsx', retrain)
        return ans


from flask import Flask, render_template, request

app = Flask(__name__, static_folder="static", template_folder="templates")


# @app.route('/predict', methods=["GET"])
# def showpredict():
#     return render_template('./line_data_pred_pic.html')


@app.route('/', methods=["GET"])
def showRank():
    return render_template("base_chart.html")


@app.route('/reflesh', methods=["POST"])
def rerun(command=''):
    data = {}
    if request:
        for key in request.values:
            data[key] = request.values.get(key)
        if request.values.get('command'):
            command = data['command']
    # if request.method == "POST":
    gradedir = './res/grade'
    resultdir = './res/result'
    ans = process_grade(gradedir, resultdir, bool(command))
    return {'return_code': '200', 'return_info': '处理成功', 'result': str(len(ans)) + '个模型预测评分已重新绘制'}
    print('完成本次预测. ')


@app.route('/line', methods=["GET"])
def dumpline(command=''):
    # if request.method == "POST":
    return ans['line_pred'].dump_options_with_quotes()


@app.route('/bar_score', methods=["GET"])
def dumpbarscore(command=''):
    return ans['bar_score'].dump_options_with_quotes()


@app.route('/bar_rmse', methods=["GET"])
def dumpbarrmse(command=''):
    return ans['bar_rmse'].dump_options_with_quotes()


@app.route('/bar_mae', methods=["GET"])
def dumpbarmae(command=''):
    return ans['bar_mae'].dump_options_with_quotes()


@app.route('/bar_acc', methods=["GET"])
def dumpbaracc(command=''):
    return ans['bar_accuracy'].dump_options_with_quotes()


@app.route('/bar_precision', methods=["GET"])
def dumpbarprec(command=''):
    return ans['bar_precision'].dump_options_with_quotes()


@app.route('/scatter', methods=["GET"])
def dumpscatter(command=''):
    return ans['scatter_predict'].dump_options_with_quotes()


'''
主函数需要手动运行
'''
if __name__ == '__main__':
    rerun()
    app.config.setdefault('BOOTSTRAP_SERVE_LOCAL', True)
    app.run(debug=1)
