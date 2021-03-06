"""
1.数据处理
"""
import numpy as np
import pandas as pd
import xlrd

if __name__ == '__main__':
    gradedir = './res/grade'
    resultdir = './res/result'
    colnames = ('NO_', 'COURE_NAME', 'ZHSCORE')
    filename = 'train.xls'
    book = xlrd.open_workbook(filename, 'r')
    sheet = book.sheet_by_index(0)
    result = dict()
    for i in range(sheet.ncols):
        if sheet.cell_value(0,i) in colnames:
            result[sheet.cell_value(0,i)] = sheet.col_values(i,1)
    result = pd.DataFrame(result)
    #替换字符串空值为nan
    result = result.replace('', np.nan)
    result = result.dropna()
    # 以学号作索引、科目名作列，分数值作行，填充na = 0
    result = result.pivot_table(index=colnames[0], columns=colnames[1], values=colnames[2], fill_value=0)
    t = result[result > 0].count()
    idx = t[t < len(result)/2].index
    result = result.drop(idx, 1)
    result[result > 100] = result - 60
    print(result)