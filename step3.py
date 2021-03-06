from process import *
from sklearn.svm import SVR,SVC
from sklearn import linear_model
import matplotlib.pyplot as plt


preGrade = read_excel('train.xls')
followGrade = read_excel('test.xls')
x_train, y_train, x_test, y_test, x_pred = get_table(preGrade, followGrade)

# 选择线性回归
# linear_svr = SVR(C=0.8, kernel='linear')
# linear_svr.fit(x_train.values, y_train.values[:,0])
# linear_svr_y_predict = linear_svr.predict(x_test.values)
# plt.figure()
# plt.plot(linear_svr_y_predict)
for i in range(len(y_train.columns)):
    linear_svr = SVR(C=0.7, kernel='linear')
    linear_svr.fit(x_train.values, y_train.values[:,i])
    y_predict = linear_svr.predict(x_pred.values)
    # 使用线性回归
    regr = linear_model.LinearRegression()
    # 进行training set和test set的fit，即是训练的过程
    # regr.fit(x_train.values,y_train.values)
    # print(regr.score(x_test.values,y_test.values))
    # y_predict = regr.predict(x_pred)
    plt.figure()
    plt.plot(y_predict)
    y_test[y_train.columns[i]] = y_predict

y_test = y_test.astype(int)
y_test[y_test > 100] = 99
y_test[y_test < 0] = 0

y_test.to_csv('y_prediction.csv', encoding='gbk')

# poly_svr = SVR(C=0.8, kernel='poly')
# poly_svr.fit(x_train.values, y_train.values[:,1])
# poly_svr_y_predict = poly_svr.predict(x_test.values)
# plt.figure()
# plt.plot(poly_svr_y_predict)
#
# rbf_svr = SVR(C=0.8, kernel='rbf')
# rbf_svr.fit(x_train.values, y_train.values[:,1])
# rbf_svr_y_predict = rbf_svr.predict(x_test.values)
# plt.figure()
# plt.plot(rbf_svr_y_predict)
