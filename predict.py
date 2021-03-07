# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
from sklearn.svm import SVR
from sklearn.linear_model import LogisticRegression, Lasso
from sklearn.ensemble import AdaBoostClassifier
import lightgbm as lgbm
from sklearn.model_selection import train_test_split
import xlrd
import matplotlib.pyplot as plt
import os

def predictLasso(x_train, y_train, x_test, y_test, x_pred):

    y_pred = pd.DataFrame(index=x_pred.index)
    y_score = pd.DataFrame(index=x_pred.index)
    for i in range(len(y_train.columns)):
        lasm = Lasso(alpha = 0.018)
        lasm.fit(x_train.values, y_train.values[:, i].astype(int))
        y_predict_svc = lasm.predict(x_pred.values)
        y_pred[y_train.columns[i]] = y_predict_svc
        y_score[y_train.columns[i]] = lasm.score(x_test.values, y_test.values[:,i].astype(int))
    y_pred = y_pred.astype(int)
    y_pred[y_pred > 100] = 99
    y_pred[y_pred < 0] = 0
    return y_pred, y_score.values[0].mean(axis=0)

def predictAdaBoost(x_train, y_train, x_test, y_test, x_pred):

    y_pred = pd.DataFrame(index=x_pred.index)
    y_score = pd.DataFrame(index=x_pred.index)
    for i in range(len(y_train.columns)):
        clf = AdaBoostClassifier(n_estimators=100)
        clf.fit(x_train.values, y_train.values[:, i].astype(int))
        y_predict_svc = clf.predict(x_pred.values)
        y_pred[y_train.columns[i]] = y_predict_svc
        y_score[y_train.columns[i]] = clf.score(x_test.values, y_test.values[:,i].astype(int))
    y_pred = y_pred.astype(int)
    y_pred[y_pred > 100] = 99
    y_pred[y_pred < 0] = 0
    return y_pred, y_score.values[0].mean(axis=0)

def predictLR(x_train, y_train, x_test, y_test, x_pred):
    y_pred = pd.DataFrame(index=x_pred.index)
    y_score = pd.DataFrame(index=x_pred.index)
    for i in range(len(y_train.columns)):
        linear = LogisticRegression(C=0.0092, tol=0.0005)
        linear.fit(x_train.values, y_train.values[:, i].astype(int))
        y_predict_svc = linear.predict(x_pred.values)
        y_pred[y_train.columns[i]] = y_predict_svc
        y_score[y_train.columns[i]] = linear.score(x_test.values, y_test.values[:,i].astype(int))
    y_pred = y_pred.astype(int)
    y_pred[y_pred > 100] = 99
    y_pred[y_pred < 0] = 0
    return y_pred, y_score.values[0].mean(axis=0)

def predictSVR(x_train, y_train, x_test, y_test, x_pred):
    y_pred = pd.DataFrame(index=x_pred.index)
    y_score = pd.DataFrame(index=x_pred.index)
    for i in range(len(y_train.columns)):
        linear_svr = SVR(C=0.7, kernel='linear')
        a = x_train.values
        b = y_train.values[:, i]
        linear_svr.fit(x_train.values, y_train.values[:, i].astype(int))
        y_predict = linear_svr.predict(x_pred.values)
        y_pred[y_train.columns[i]] = y_predict
        y_score[y_train.columns[i]] = linear_svr.score(x_test.values, y_test.values[:,i].astype(int))
    y_pred = y_pred.astype(int)
    y_pred[y_pred > 100] = 99
    y_pred[y_pred < 0] = 0
    a = y_score.values[0].mean(axis=0)
    return y_pred, a

def predictLGBM(x_train, y_train, x_test, y_test, x_pred):
    y_pred = pd.DataFrame(index=x_pred.index)
    y_score = pd.DataFrame(index=x_pred.index)
    for i in range(len(y_train.columns)):
        GBMmodel = lgbm.LGBMRegressor(
            objective='regression_l1',
            max_depth=5,
            num_leaves=15,
            learning_rate=0.01,
            n_estimators=1000,
            min_child_samples=80,
            subsample=0.8,
            colsample_bytree=1,
            reg_alpha=0,
            reg_lambda=0,
            random_state=np.random.randint(10e6))
        GBMmodel.fit(
            x_train.values,
            y_train.values[:, i],
            eval_set=[(y_train.values, y_train.values[:, i].astype(int)), (x_test.values, y_test.values[:, i].astype(int))],
            eval_names=('fit', 'val'),
            eval_metric='l1',
            # early_stopping_rounds=200,
            verbose=False)
        y_predict_GBM = GBMmodel.predict(x_pred.values)
        y_pred[y_train.columns[i]] = y_predict_GBM
        y_score[y_train.columns[i]] = GBMmodel.score(x_test.values, y_test.values
                                                     [:, i].astype(int))
    y_pred = y_pred.astype(int)
    y_pred[y_pred > 100] = 99
    y_pred[y_pred < 0] = 0
    return y_pred, y_score.values[0].mean(axis=0)