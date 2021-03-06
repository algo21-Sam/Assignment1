# coding:utf-8 
"""
author:Sam
date：2021/2/2
"""

import numpy as np
import pandas as pd
import scipy.io
from scipy import stats
import time
import math
import matplotlib.pyplot as plt
import statsmodels.api as sm
from collections import Counter

# 计时开始
time_start = time.time()


# 第一步：数据读取/输入
# 1.1 mat文件数据读取（已有因子值）
DayPrice_stock = scipy.io.loadmat('D:\\pycharm project\\factor_testing\\DayPrice_Stock.mat')                      # 输入因子值文件
factor = scipy.io.loadmat('D:\\pycharm project\\factor_testing\\MV.mat')      # 输入因子值文件
industry =  scipy.io.loadmat('D:\\pycharm project\\factor_testing\\CSLv1.mat')

# 1.2 因子值计算
data1=scipy.io.loadmat('NPParentCompanyOwners.mat')
time=data1['Report_dates_num'].reshape(1,-1)[0]
time=pd.to_datetime(time-719529, unit='D')
stockName=[]
for i in data1['RICs'].reshape(1,-1)[0]:
    stockName.append(i[0])
profit=pd.DataFrame(data1['NPParentCompanyOwners_TTM'],index=time,columns=stockName)



# 第二步：数据预处理
# 2.1 mat格式数据处理
# 将array序列转为list
def array_to_list(factor,column):
    list=[]
    for x in factor[column].reshape(1,-1)[0]:
        x[0] = str(x[0])  # 这里再研究一下
        list.append(x[0])
    return list

# 将array序列转为datetime
def array_to_datetime(factor,column):
    x = pd.to_datetime(pd.Series(factor[column].flatten()).apply(lambda x:str(x[0])))
    return x

# 将date_num转化为list
def datenum_to_list(factor):
    date_num = []
    for x in factor['date_num']:
        date_num.append(x[0])
    return date_num


# 2.2 统一日期 日期对齐
"""     datetime number
DayPrice_stock: 731220-737859 日度数据(4409)
factor: 
industry: 731583-737860 日度数据（4173）
"""
# date_num 和 date 对应关系
a = array_to_datetime(DayPrice_stock,'date')
b = datenum_to_list(DayPrice_stock)
date = pd.DataFrame(a.values,index = b,columns=['date'])
print(date)

# 取三个表的日期交集
start_date_num = max(b[0],datenum_to_list(factor)[0],datenum_to_list(industry)[0])
end_date_num = min(b[-1],datenum_to_list(factor)[-1],datenum_to_list(industry)[-1])
date = date.loc[start_date_num:end_date_num]


#将季度报日期转为证监会规定的财报披露的最后一天。输入季频数据，输出季频数据
def statementDatesDeal(data):
    dates=data.index
    datesNew=[]
    #由于年报与一季度报的披露最晚日期都是4/30，我们删除较旧的12/31的数据
    for date in dates[:-1]:
        d=str(date)[:10]
        if d[5:]=="03-31":
            datesNew.append(d[:5]+"4-30")
        if d[5:]=="06-30":
            datesNew.append(d[:5]+"8-30")
        if d[5:]=="09-30":
            datesNew.append(d[:5]+"10-31")
        if d[5:]=="12-31":
            data=data.drop(date,axis=0)
    #最后一天如果是12/31，则不用删，用其作为第二年
    for date in [dates[-1]]:
        d=str(date)[:10]
        if d[5:]=="03-31":
            datesNew.append(d[:5]+"4-30")
        if d[5:]=="06-30":
            datesNew.append(d[:5]+"8-30")
        if d[5:]=="09-30":
            datesNew.append(d[:5]+"10-31")
        if d[5:]=="12-31":
            datesNew.append(str(int(d[:4])+1)+"-"+"4-30")
    data.index=pd.to_datetime(datesNew)
    return data

#季度频率变日频
def to_daily_freq(quarter,trade_date):
    dates_q = pd.to_datetime(quarter.index)
    dates_d = pd.to_datetime(trade_date)
    df = pd.DataFrame(None,index=dates_d,columns=quarter.columns)
    count = 0
    for i in df.index:
        if (i>= dates_q[count]) & (i< dates_q[count+1]):
            df.loc[i] = quarter.iloc[count]

        else:
            count = count+1
            df.loc[i] = quarter.iloc[count]

    print(df)
    return df

print(statementDatesDeal(profit[10:]))
df = to_daily_freq(statementDatesDeal(profit[10:]),date['date'])
print(df)
# df.to_csv('tempt.csv')






