# -*- coding: utf-8 -*-
import os
import joblib
import lightgbm as lgbm
import numpy as np
import pandas as pd
from sklearn.ensemble import AdaBoostClassifier, AdaBoostRegressor
from sklearn.linear_model import LogisticRegression, Lasso
from sklearn.model_selection import GridSearchCV
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVR, SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import r2_score, f1_score, classification_report, accuracy_score,precision_score

PRe = ['precision', 'recall']
# RG

def predictLasso(x_train, y_train, x_test, y_test, x_pred, retrain):
    y_pred = pd.DataFrame(index=x_pred.index)
    y_score = pd.DataFrame(index=x_pred.index)
    y_mae = pd.DataFrame(index=x_pred.index)
    y_rmse = pd.DataFrame(index=x_pred.index)
    for i in range(len(y_train.columns)):
        model, initialized = readModels(y_train.columns[i], 'Regression/Lasso')
        if initialized or retrain:
            model = Lasso(alpha=0.018)
            model.fit(x_train.values, y_train.values[:, i].astype(int))
        y_predict = model.predict(x_pred.values)
        y_pred[y_train.columns[i]] = y_predict
        y_predict_Test = model.predict(x_test.values)
        # 误差标准
        y_mae[y_train.columns[i]], y_rmse[y_train.columns[i]] = calculateRScore(y_predict_Test,y_test.values[:, i].astype(int))
        #R^2评分
        y_score[y_train.columns[i]] = model.score(x_test.values, y_test.values[:, i].astype(int))
        saveModels(model, 'Regression/Lasso', y_train.columns[i])
    y_pred = y_pred.astype(int)
    y_pred[y_pred > 100] = 99
    y_pred[y_pred < 0] = 0
    return y_pred, y_score.values[0].mean(axis=0), y_mae.values[0].mean(axis=0), y_rmse.values[0].mean(axis=0)

#RG
def predictAdaBoostR(x_train, y_train, x_test, y_test, x_pred, retrain):
    y_pred = pd.DataFrame(index=x_pred.index)
    y_score = pd.DataFrame(index=x_pred.index)
    y_mae = pd.DataFrame(index=x_pred.index)
    y_rmse = pd.DataFrame(index=x_pred.index)
    for i in range(len(y_train.columns)):
        model, initialized = readModels(y_train.columns[i], 'Regression/AdaBoost')
        if initialized or retrain:
            model = AdaBoostRegressor(n_estimators=50, learning_rate=0.1, loss='linear')
            model.fit(x_train.values, y_train.values[:, i].astype(int))
        y_predict = model.predict(x_pred.values)
        y_pred[y_train.columns[i]] = y_predict
        y_predict_Test = model.predict(x_test.values)
        # 误差标准
        y_mae[y_train.columns[i]], y_rmse[y_train.columns[i]] = calculateRScore(y_predict_Test,y_test.values[:, i].astype(int))
        # R^2评分
        y_score[y_train.columns[i]] = model.score(x_test.values, y_test.values[:, i].astype(int))
        saveModels(model, 'Regression/AdaBoost', y_train.columns[i])
    y_pred = y_pred.astype(int)
    y_pred[y_pred > 100] = 99
    y_pred[y_pred < 0] = 0
    return y_pred, y_score.values[0].mean(axis=0), y_mae.values[0].mean(axis=0), y_rmse.values[0].mean(axis=0)


# 创建文件目录
def saveModels(model, classes, modelname):
    dirs = './models/' + classes
    if not os.path.exists(dirs):
        os.makedirs(dirs)
    path = dirs + '/' + modelname + '.pkl'
    joblib.dump(model, path)
    return path

#读取对应模型
def readModels(modelname, classes):
    try:
        path = './models/' + classes + '/' + modelname + '.pkl'
        clf = joblib.load(path)
        return clf, False
    except Exception as e:
        print("读取模型失败 : ", e)
        return None, True
#计算指标rmse and rae
def calculateRScore(y_predict,y_test):
    return np.mean(abs(y_predict - y_test)), np.sqrt(np.mean((y_predict - y_test) ** 2))
#计算指标准确率
def calculateCScore(y_predict,y_test):
    return precision_score(y_test, y_predict, average='macro'), accuracy_score(y_test,y_predict)

#
def predictAdaBoost(x_train, y_train, x_test, y_test, x_pred, retrain):
    # tuned_parameters = [
    #                     {'n_estimators': range(10,100,10), 'algorithm': ['SAMME.R']}
    #                     ]
    y_pred = pd.DataFrame(index=x_pred.index)
    y_score = pd.DataFrame(index=x_pred.index)
    y_precision = pd.DataFrame(index=x_pred.index)
    y_accuracy = pd.DataFrame(index=x_pred.index)
    for i in range(len(y_train.columns)):
        model, initialized = readModels(y_train.columns[i], 'Classification/AdaBoost')
        if initialized or retrain:
            model = AdaBoostClassifier(n_estimators=90, learning_rate=0.15, algorithm='SAMME.R')
            model.fit(x_train.values, y_train.values[:, i].astype(int))
        # for s in PRe:
        #     searchCV = GridSearchCV(model, param_grid=tuned_parameters, cv=5,
        #                     scoring='%s_macro' % s)
        #     searchCV.fit(x_train.values, y_train.values[:, i].astype(int))
        #     print('best params:', searchCV.best_params_)
        #     print('best score:', searchCV.best_score_)
        #     y_preda = searchCV.predict(x_test)
        #     print(classification_report(y_test.values[:,i].astype(int), y_preda ))
        y_predict = model.predict(x_pred.values)
        y_pred[y_train.columns[i]] = y_predict
        y_pred_Test = model.predict(x_test.values)
        y_precision[y_train.columns[i]], y_accuracy[y_train.columns[i]] = calculateCScore(y_pred_Test, y_test.values[:, i].astype(int))
        y_score[y_train.columns[i]] = f1_score(y_test.values[:, i].astype(int), y_pred_Test, average='macro')
        # y_score[y_train.columns[i]] = model.score(x_test.values, y_test.values[:, i].astype(int))
        saveModels(model, 'Classification/AdaBoost', y_train.columns[i])
    y_pred = y_pred.astype(int)
    y_pred[y_pred > 100] = 99
    y_pred[y_pred < 0] = 0
    return y_pred, y_score.values[0].mean(axis=0), y_precision.values[0].mean(axis=0), y_accuracy.values[0].mean(axis=0)


def predictLR(x_train, y_train, x_test, y_test, x_pred, retrain):
    y_pred = pd.DataFrame(index=x_pred.index)
    y_score = pd.DataFrame(index=x_pred.index)
    y_precision = pd.DataFrame(index=x_pred.index)
    y_accuracy = pd.DataFrame(index=x_pred.index)
    for i in range(len(y_train.columns)):
        model, initialized = readModels(y_train.columns[i], 'Classification/LogicRegression')
        if initialized or retrain:
            model = LogisticRegression(C=0.0092, tol=0.0005)
            model.fit(x_train.values, y_train.values[:, i].astype(int))
        y_predict = model.predict(x_pred.values)
        y_pred[y_train.columns[i]] = y_predict
        y_pred_Test = model.predict(x_test.values)
        y_precision[y_train.columns[i]], y_accuracy[y_train.columns[i]] = calculateCScore(y_pred_Test,y_test.values[:, i].astype(int))
        y_score[y_train.columns[i]] = f1_score(y_test.values[:, i].astype(int), y_pred_Test, average='macro')
        # y_score[y_train.columns[i]] = model.score(x_test.values, y_test.values[:, i].astype(int))
        saveModels(model, 'Classification/LogicRegression', y_train.columns[i]), y_precision.values[0].mean(axis=0), y_accuracy.values[0].mean(axis=0)
    y_pred = y_pred.astype(int)
    y_pred[y_pred > 100] = 99
    y_pred[y_pred < 0] = 0
    return y_pred, y_score.values[0].mean(axis=0), y_precision.values[0].mean(axis=0), y_accuracy.values[0].mean(axis=0)


# said
def predictSVR(x_train, y_train, x_test, y_test, x_pred, retrain):
    # tuned_parameters = [
    #                     {'kernel': ['rbf'], 'gamma': ['auto', 'scale'],
    #                      # 'C': [0.1, 0.3, 0.5, 0.7, 1], 'shrinking': [True, False],
    #                      },
    #                     {'kernel': ['linear'], 'epsilon': [0.1, 0.3, 0.5, 0.7, 1]},
    #                     ]
    y_pred = pd.DataFrame(index=x_pred.index)
    y_score = pd.DataFrame(index=x_pred.index)
    y_mae = pd.DataFrame(index=x_pred.index)
    y_rmse = pd.DataFrame(index=x_pred.index)
    for i in range(len(y_train.columns)):
        model, initialized = readModels(y_train.columns[i], 'Regression/SVR')
        if initialized or retrain:
            model = SVR(C=0.62, kernel='linear', epsilon=1)
            model.fit(x_train.values, y_train.values[:, i].astype(int))
        # searchCV = d(model,
        #                         param_grid=tuned_parameters,
        #                         cv=5,
        #                         scoring='neg_mean_squared_error',
        #                         # verbose=1
        #                         )
        # searchCV.fit(x_train.values, y_train.values[:, i].astype(int))
        # print('best params:', searchCV.best_params_)
        # print('best score:', searchCV.best_score_)
        # y_preda = searchCV.score(x_test.values, y_test.values[:, i].astype(int))
        # print(y_preda)
        y_predict = model.predict(x_pred.values)
        y_pred[y_train.columns[i]] = y_predict
        # 误差标准
        y_predict_Test = model.predict(x_test.values)
        y_mae[y_train.columns[i]], y_rmse[y_train.columns[i]] = calculateRScore(y_predict_Test,y_test.values[:, i].astype(int))
        # R^2评分
        y_score[y_train.columns[i]] = model.score(x_test.values, y_test.values[:, i].astype(int))
        saveModels(model, 'Regression/SVR', y_train.columns[i])
    y_pred = y_pred.astype(int)
    y_pred[y_pred > 100] = 99
    y_pred[y_pred < 0] = 0
    return y_pred, y_score.values[0].mean(axis=0), y_mae.values[0].mean(axis=0), y_rmse.values[0].mean(axis=0)


def predictSVC(x_train, y_train, x_test, y_test, x_pred, retrain):
    # tuned_parameters = [
    #                     {'kernel': ['linear'], 'C': range(10,30,1)}
    #                     ]
    y_pred = pd.DataFrame(index=x_pred.index)
    y_score = pd.DataFrame(index=x_pred.index)
    y_precision = pd.DataFrame(index=x_pred.index)
    y_accuracy = pd.DataFrame(index=x_pred.index)
    for i in range(len(y_train.columns)):
        model, initialized = readModels(y_train.columns[i], 'Classification/SVC')
        if initialized or retrain:
            model = SVC(kernel='linear', C=11)
            model.fit(x_train.values, y_train.values[:, i].astype(int))
        # for s in PRe:
        #     searchCV = GridSearchCV(model, param_grid=tuned_parameters, cv=5,
        #                     scoring='%s_macro' % s)
        #     searchCV.fit(x_train.values, y_train.values[:, i].astype(int))
        #     print('best params:', searchCV.best_params_)
        #     print('best score:', searchCV.best_score_)
        #     y_preda = searchCV.predict(x_test)
        #     print(classification_report(y_test.values[:,i].astype(int), y_preda ))
        y_predict = model.predict(x_pred.values)
        y_pred[y_train.columns[i]] = y_predict
        y_pred_Test = model.predict(x_test.values)
        y_precision[y_train.columns[i]], y_accuracy[y_train.columns[i]] = calculateCScore(y_pred_Test,y_test.values[:, i].astype(int))
        y_score[y_train.columns[i]] = f1_score(y_test.values[:, i].astype(int), y_pred_Test, average='macro')
        # y_score[y_train.columns[i]] = model.score(x_test.values, y_test.values[:, i].astype(int))
        saveModels(model, 'Classification/SVC', y_train.columns[i])
    y_pred = y_pred.astype(int)
    y_pred[y_pred > 100] = 99
    y_pred[y_pred < 0] = 0
    return y_pred, y_score.values[0].mean(axis=0), y_precision.values[0].mean(axis=0), y_accuracy.values[0].mean(axis=0)


def predictNB(x_train, y_train, x_test, y_test, x_pred, retrain):
    y_pred = pd.DataFrame(index=x_pred.index)
    y_score = pd.DataFrame(index=x_pred.index)
    y_precision = pd.DataFrame(index=x_pred.index)
    y_accuracy = pd.DataFrame(index=x_pred.index)
    for i in range(len(y_train.columns)):
        model, initialized = readModels(y_train.columns[i], 'Classification/Bayes')
        if initialized or retrain:
            model = GaussianNB()
        # 训练模型+预测数据
            model.fit(x_train.values, y_train.values[:, i].astype(int))
        y_predict = model.predict(x_pred.values)
        y_pred[y_train.columns[i]] = y_predict
        y_pred_Test = model.predict(x_test.values)
        y_precision[y_train.columns[i]], y_accuracy[y_train.columns[i]] = calculateCScore(y_pred_Test, y_test.values[:, i].astype(int))
        y_score[y_train.columns[i]] = f1_score(y_test.values[:, i].astype(int), y_pred_Test, average='macro')
        # y_score[y_train.columns[i]] = model.score(x_test.values, y_test.values[:, i].astype(int))
        saveModels(model, 'Classification/Bayes', y_train.columns[i])
    y_pred = y_pred.astype(int)
    y_pred[y_pred > 100] = 99
    y_pred[y_pred < 0] = 0
    return y_pred, y_score.values[0].mean(axis=0), y_precision.values[0].mean(axis=0), y_accuracy.values[0].mean(axis=0)


def predictLGBM(x_train, y_train, x_test, y_test, x_pred, retrain):
    y_pred = pd.DataFrame(index=x_pred.index)
    y_score = pd.DataFrame(index=x_pred.index)
    y_mae = pd.DataFrame(index=x_pred.index)
    y_rmse = pd.DataFrame(index=x_pred.index)
    boostRound = 100
    earlyStopRound = 30
    for i in range(len(y_train.columns)):
        # lgb_train = lgbm.Dataset(x_train.values,
        #                          y_train.values[:, i])
        # params = {
        #     'boosting_type': 'gbdt',
        #     'objective': 'regression_l2',
        #     'task': 'predict',
        #     'force_col_wise': True,
        #     'metric': [
        #         'rmse',
        #         'loss'],
        #     'max_depth': 3,
        #     'num_leaves': 7,
        #     'learning_rate': 0.05,
        #     'feature_fraction': 0.8,
        #     'bagging_fraction': 0.7,
        #     'bagging_freq': 2,
        #     'verbose': 1
        # }
        # cv_results = lgbm.cv(
        #     params, lgb_train, num_boost_round=1000, nfold=5, stratified=False, shuffle=True, metrics=[
        #         'rmse',
        #         'loss'],
        #     early_stopping_rounds=50, verbose_eval=50, show_stdv=True, seed=0)
        # print('best n_estimators:', len(cv_results['rmse-mean']))
        # print('best cv score:', cv_results['rmse-mean'][-1])
        model, initialized = readModels(y_train.columns[i], 'Regression/LightGBM')
        if initialized or retrain:
            lgb_train = lgbm.Dataset(x_train.values,
                                     y_train.values[:, i])
            lgb_eval = lgbm.Dataset(x_test.values, y_test.values[:, i], reference=lgb_train)
            # params = {
            #     'boosting_type': 'gbdt',
            #     'objective': 'regression',
            #     'task': 'predict',
            #     'force_col_wise': True,
            #     'metric': [
            #         'rmse',
            #         'loss'],
            #     'max_depth': 6,
            #     'num_leaves': 62,
            #     'learning_rate': 0.05,
            #     'feature_fraction': 0.8,
            #     'bagging_fraction': 0.7,
            #     'bagging_freq': 2,
            #     'verbose': 1
            # }
            # res = {}
            # model = lgbm.train(
            #     params,
            #     lgb_train,
            #     num_boost_round=boostRound,
            #     valid_sets=(lgb_eval, lgb_train),
            #     valid_names=('validate', 'train'),
            #     early_stopping_rounds=earlyStopRound,
            #     evals_result= res,
            # )
            model = lgbm.LGBMRegressor(
                boosting_type='gbdt',
                objective='regression',
                max_depth=6,
                metric=
                [
                    'rmse',
                    'loss'],
                num_leaves=52,
                learning_rate=0.1,
                n_estimators=116,
                bagging_fraction=0.8,
                feature_fraction=0.7,
                task='predict',
                # min_child_samples=80,
                # reg_alpha=0,
                # reg_lambda=0,
                # random_state=np.random.randint(10e6)
            )
            model.fit(
                x_train.values,
                y_train.values[:, i],
                eval_set=[(x_test.values, y_test.values[:, i].astype(int))],
                eval_names=
                (
                    'fit',
                    'val'
                ),
                eval_metric=
                [
                    'rmse',
                    'loss'],
                early_stopping_rounds=50,
                # verbose=True
            )
            # params_test1 = {
            #     # 'max_depth': range(1, 10, 2),
            #     # 'num_leaves': range(2, 160, 2)
            #     'n_estimators': range(90, 120, 2),
            #     'feature_fraction': [0.6, 0.7, 0.8, 0.9, 1],
            # }
            # gsearch1 = GridSearchCV(estimator=model, param_grid=params_test1, scoring='neg_mean_squared_error', cv=5,
            #                         verbose=1, n_jobs=4,)
            # gsearch1.fit(x_train.values,
            #     y_train.values[:, i])
            # print(gsearch1.best_params_, gsearch1.best_score_)
            # params_test2 = {
            #     'max_depth': range(6, 9, 1),
            #     'num_leaves': range(50, 80, 2)
            # }
            # gsearch2 = GridSearchCV(estimator=model, param_grid=params_test2, scoring='neg_mean_squared_error', cv=5,
            #                         verbose=1, n_jobs=4)
            # gsearch2.fit(x_train.values,
            #     y_train.values[:, i])
            # print(gsearch2.best_params_, gsearch2.best_score_)
        y_predict_GBM = model.predict(x_pred.values)
        y_pred[y_train.columns[i]] = y_predict_GBM
        y_predict_test = model.predict(x_test.values)
        # y_predict_GBM = model.predict(x_pred.values, num_iteration= model.best_iteration)
        # y_pred[y_train.columns[i]] = y_predict_GBM
        # y_predict_test = model.predict(x_test.values, num_iteration= model.best_iteration)
        auc = r2_score(y_test.values[:, i], y_predict_test)
        # auc =model.best_score.get('validate')['auc']
        score_array = np.empty(y_score.shape[0])
        score_array.fill(auc)
        # 误差标准
        y_mae[y_train.columns[i]], y_rmse[y_train.columns[i]] = calculateRScore(y_predict_test,y_test.values[:, i].astype(int))
        # R^2评分
        y_score[y_train.columns[i]] = score_array
        saveModels(model, 'Regression/LightGBM', y_train.columns[i])
    y_pred = y_pred.astype(int)
    y_pred[y_pred > 100] = 99
    y_pred[y_pred < 0] = 0
    return y_pred, y_score.values[0].mean(axis=0), y_mae.values[0].mean(axis=0), y_rmse.values[0].mean(axis=0)


def predictDT(x_train, y_train, x_test, y_test, x_pred, retrain):
    y_pred = pd.DataFrame(index=x_pred.index)
    y_score = pd.DataFrame(index=x_pred.index)
    y_precision = pd.DataFrame(index=x_pred.index)
    y_accuracy = pd.DataFrame(index=x_pred.index)
    for i in range(len(y_train.columns)):
        model, initialized = readModels(y_train.columns[i], 'Classification/DecisionTree')
        if initialized or retrain:
            model = DecisionTreeClassifier()
            model.fit(x_train.values, y_train.values[:, i].astype(int))
        y_predict_model = model.predict(x_pred.values)
        y_pred[y_train.columns[i]] = y_predict_model
        y_pred_Test = model.predict(x_test.values)
        y_precision[y_train.columns[i]], y_accuracy[y_train.columns[i]] = calculateCScore(y_pred_Test, y_test.values[:, i].astype(int))
        y_score[y_train.columns[i]] = f1_score(y_test.values[:, i].astype(int), y_pred_Test, average='macro')
        # y_score[y_train.columns[i]] = model.score(x_test.values, y_test.values[:, i].astype(int))
        saveModels(model, 'Classification/DecisionTree', y_train.columns[i])
    y_pred = y_pred.astype(int)
    y_pred[y_pred > 100] = 99
    y_pred[y_pred < 0] = 0
    return y_pred, y_score.values[0].mean(axis=0), y_precision.values[0].mean(axis=0), y_accuracy.values[0].mean(axis=0)