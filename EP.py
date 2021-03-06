# coding:utf-8 
"""
author:Sam
date：2021/2/1
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
# time_start = time.time()


# 第一步：数据读取/输入
# 1.1 mat文件数据读取（已有因子值）
DayPrice_stock = scipy.io.loadmat('D:\\pycharm project\\factor_testing\\DayPrice_Stock.mat')                      # 输入因子值文件
factor1 = scipy.io.loadmat('D:\\pycharm project\\factor_testing\\MV.mat')      # 输入因子值文件
industry =  scipy.io.loadmat('D:\\pycharm project\\factor_testing\\CSLv1.mat')

# 1.2 CSV文件数据读取（已有因子值）
# DayPrice_stock = pd.read_csv('DayPrice_Stock.csv')
# factor = pd.read_csv('D:\\pycharm project\\factor_testing\\factor.csv')      # 输入因子值文件
# industry =  pd.read_csv('CSLv1.csv')

# 1.3 因子值计算（未有因子值，需要通过计算得到）
# 归母净利润TTM（季频转日频）
factor2=scipy.io.loadmat('NPParentCompanyOwners.mat')
time=factor2['Report_dates_num'].reshape(1,-1)[0]
time=pd.to_datetime(time-719529, unit='D')
stockName=[]
for i in factor2['RICs'].reshape(1,-1)[0]:
    stockName.append(i[0])
profit=pd.DataFrame(factor2['NPParentCompanyOwners_TTM'],index=time,columns=stockName)

# 将季度报日期转为证监会规定的财报披露的最后一天。输入季频数据，输出季频数据
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
    return df




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
# print(date)

# 取三个表的日期交集
# start_date_num = max(b[0],datenum_to_list(factor1)[0],datenum_to_list(industry)[0])
start_date_num = 734142
end_date_num = min(b[-1],datenum_to_list(factor1)[-1],datenum_to_list(industry)[-1])
date = date.loc[start_date_num:end_date_num]
# print(date)


# 2.3
# 获取股票池收盘价数据
def stock(DayPrice_stock):
    stockName = []
    for i in DayPrice_stock['RICs'].reshape(1, -1)[0]:
        stockName.append(i[0])
    price = DayPrice_stock['ClosePrice']
    time = array_to_datetime(DayPrice_stock,'date')
    price = pd.DataFrame(price, columns=stockName, index=time)
    return price


# 2.4 临时处理函数
# series删除0
def drop_zero(series):
    dict = {}
    for x,y in series.items():
        if y == 0:
            continue
        else:
            dict['x'] = y
    new_series = pd.Series(dict)
    return new_series


# 第三步：
# 3.1 异常值处理：MAD中位数去极值法
def filter_extreme_MAD(series,n=3*1.4826):
    median = series.median()
    new_median = ((series - median).abs()).median()
    max_range = median + n*new_median
    min_range = median - n*new_median
    return np.clip(series,min_range,max_range)

# 3.2 行业中性化
# 获取当日行业标签
def get_industry_exposure(stock_list,industry,date):
    df = pd.DataFrame(index=range(1001,1030), columns=stock_list)
    for stock in stock_list:
        label = industry.loc[date][stock]
        try:
            df.loc[label][stock] = 1
        except:
            continue
    return df.fillna(0)  # 将NaN赋为0

# 行业中性
# def neutralization(series,industry_label):
#     date = series.name
#     date = date.strftime('%Y-%m-%d')
#     index = series.dropna().index
#     y = series.dropna()
#     # 这里开始有问题
#     dummy_var = pd.get_dummies(industry_label.loc[date]).drop([0], axis=1)
#     # dummy_var = get_industry_exposure(industry_label.columns, industry_label, date).T
#     x = dummy_var.loc[index]
#     result = sm.OLS(y.astype(float), x.astype(float)).fit()
#     residue = result.resid
#     return residue

# 中性化（行业中性+市值中性）
def neutralization(series,industry_label,market_value):
    date = series.name
    date = date.strftime('%Y-%m-%d')
    index = series.dropna().index
    y = series.dropna()
    # print(len(y))
    # 这里开始有问题
    dummy_var = pd.get_dummies(industry_label.loc[date]).drop([0], axis=1)
    # dummy_var = get_industry_exposure(industry_label.columns, industry_label, date).T
    ind_dummy = dummy_var.loc[index]
    # print(len(ind_dummy))

    # 对齐处理
    mkt_value = market_value.loc[date]
    mkt_value = mkt_value[index]
    # print(len(mkt_value))
    # 原先的做法 先保留
    # mkt_value = market_value.loc[date].replace(0,np.nan)
    # mkt_value = mkt_value.dropna()
    # print(len(mkt_value))

    x = pd.concat([mkt_value,ind_dummy],axis=1)   # 市值中性化+行业中性化
    x[np.isnan(x)] = 0
    x[np.isinf(x)] = 0
    result = sm.OLS(y.astype(float), x.astype(float)).fit()
    residue = result.resid
    return residue


# # 重新填充中性化后的因子值表（行业中性化）
# def set_new_factor_df(standardize_factor,industry_label):
#     d = []
#     for date in standardize_factor.index:
#         date = date.strftime('%Y-%m-%d')
#         stock_index = standardize_factor.loc[date].dropna().index
#         initList = [[np.nan for x in range(len(standardize_factor.columns))]]
#         temp = pd.DataFrame(initList, columns=standardize_factor.columns)
#         series = neutralization(standardize_factor.loc[date], industry_label)
#         temp[stock_index] = [series.values]
#         d.append(temp.values[0])
#     result = pd.DataFrame(d,index = standardize_factor.index, columns=standardize_factor.columns)
#     return result

# 重新填充中性化后的因子值表（行业+市值中性化）
def set_new_factor_df(standardize_factor,industry_label,market_value):
    d = []
    for date in standardize_factor.index:
        date = date.strftime('%Y-%m-%d')
        stock_index = standardize_factor.loc[date].dropna().index
        initList = [[np.nan for x in range(len(standardize_factor.columns))]]
        temp = pd.DataFrame(initList, columns=standardize_factor.columns)
        series = neutralization(standardize_factor.loc[date], industry_label, market_value)
        temp[stock_index] = [series.values]
        d.append(temp.values[0])
    result = pd.DataFrame(d,index = standardize_factor.index, columns=standardize_factor.columns)
    return result



# 第四步：因子评价体系
# 4.1 IC评价体系
# 计算序列大于n的概率
def larger_than_prob(series,n):
    count = 0
    for x in series:
        if x >= n:
            count = count+1
        else:
            continue
    prob = count/len(series)
    return prob

# 计算平均年化收益率
def annul_return_am(n2,t):
    x = math.pow(math.pow(n2,1/t),240) - 1
    return x


# 计算最大回撤率(好像有点问题?)
def max_drawback(series):
    series = series.tolist()
    m_drawback = []
    for n in range(0,len(series)):
        find_max = max(series[0:n+1])
        index = series.index(find_max)
        find_min = min(series[index:n+1])
        drawback = (find_max-find_min)/find_max
        m_drawback.append(drawback)
    max_drawback = max(m_drawback)
    # print(m_drawback.index(max_drawback))
    return max_drawback

# 计算夏普比
def sharpe_ratio(series,last_value):
    std = series.std() * (240 ** 0.5)
    list = series.tolist()
    r = annul_return_am(last_value,3688)
    sharpe_ratio = r/std
    return sharpe_ratio

# 计算年化波动率
def annul_volatility(series):
    std = series.std() * (240 ** 0.5)
    return std



# 4.2 分组回溯法
# 将传入的股票池list分为5个等长度的组合
def div_group(series):
    series = series.dropna()
    # print("当前调仓日可交易股票总数量为：" + str(len(series)))
    k = len(series) % 5   #求余
    new_series = series.sort_values()[0:len(series)-k]
    num_single_group = len(new_series) / 5
    # print("每一组的长度为:" + str(num_single_group))
    new_series_index = new_series.index.to_list()
    num_single_group = int(num_single_group)
    group1 = new_series_index[0:num_single_group]                       # 因子值最小的组
    group2 = new_series_index[num_single_group:2*num_single_group]
    group3 = new_series_index[2*num_single_group:3*num_single_group]
    group4 = new_series_index[3*num_single_group:4*num_single_group]
    group5 = new_series_index[4*num_single_group:5*num_single_group]    # 因子值最大的组
    dict = {'group1':group1,
            'group2':group2,
            'group3':group3,
            'group4':group4,
            'group5':group5}
    return dict

# 挑选出月调仓日
def sel_monthly_trade_date(df,index):
    list = []
    for x in index:
        tempt_df = df.loc[x]
        first_day = tempt_df.iloc[0]
        list.append(first_day.name)
    return list

# 根据股票日收益率计算出复利月收益率
def monthly_comp_return(df,index):
    print("股票复利月收益率如下：")
    dict = {}
    for x in index:
        tempt_df = df.loc[x]
        first_day = tempt_df.iloc[0]
        for stock in tempt_df.columns:
            for n in tempt_df.index:
                first_day[stock] = first_day[stock] * (1 + tempt_df.loc[n][stock])
        dict[x] = first_day

    dict = pd.DataFrame(dict)
    return dict

# 获取输入日期对应的当月第一个交易日日期
def get_first_day(df,index):
    index = index.strftime('%Y-%m')
    tempt_df = df.loc[index]
    first_day = tempt_df.iloc[0]
    first_day_index = first_day.name
    first_day_index = first_day_index.strftime('%Y-%m-%d')
    return first_day_index




if __name__ == '__main__':
    # EP因子
    print("-------------EP因子测试开始----------")
    # 市值MV
    f_date_number = datenum_to_list(factor1)
    new_date = date
    factor1 = pd.DataFrame(factor1['MV'],columns=array_to_list(factor1,'RICs'))  # 读取MV因子值
    factor1 = factor1[0:len(new_date)]
    factor1.index = new_date['date']
    stocks=factor1.columns & stock(DayPrice_stock).columns    # 取原股票池和市值因子MV股票池的交集
    factor1=factor1[stocks]
    # print("取股票池交集后，市值因子MV的因子DateFrame如下：")
    # print(factor1)                                        # 初始因子值DataFrame
    # print("总期数："+str(len(factor1)))
    # print("股票池数量："+str(len(factor1.columns)))


    # 归母净利润TTM因子
    factor2 = to_daily_freq(statementDatesDeal(profit[10:]),date['date'])
    # print(factor2)


    # EP因子
    factor = factor2/factor1
    print(factor)


    # MAD中位数去极值
    for x in factor.index:
        factor.loc[x] = filter_extreme_MAD(factor.loc[x])
    # print("经过中性化去极值后的因子值表：")
    # print(factor)


    # Z-score标准化
    # 多加一步 把0替换成NaN
    factor = factor.replace(0, np.nan)
    standardize_factor =pd.DataFrame(stats.zscore(factor,nan_policy='omit',axis=1),columns=factor.columns,index=factor.index)
    # print("MAD去极值、Z-score标准化处理后的因子值表：")
    # print(standardize_factor)


    # 行业中性化
    industry_df = industry['CSLv1']
    industry_df = pd.DataFrame(industry_df)
    industry_date_num = datenum_to_list(industry)
    ind_start = industry_date_num.index(start_date_num) + 1
    ind_end = industry_date_num.index(end_date_num) + 2
    industry_label = industry_df.iloc[ind_start:ind_end]
    industry_label.index = standardize_factor.index
    industry_label.columns = standardize_factor.columns
    # print(industry_label)


    # 行业中性化运算
    # start = time.time()
    print("行业中性化开始")
    new_factor_df = set_new_factor_df(standardize_factor[0:100], industry_label[0:100], factor1[0:100])
    # new_factor_df.to_csv('ep_neu.csv')
    print("行业中性化结束")
    # end = time.time()



    # 直接读取行业中性化结果
    # print("行业中性化后的因子值表：")
    # new_factor_df = pd.read_csv('factor_neu.csv')
    # new_factor_df.index = pd.to_datetime(new_factor_df['date'])
    # new_factor_df = new_factor_df.drop(['date'],axis=1)
    # print(new_factor_df)


    # 对行业中性化之后的因子值表再做一次Z-score标准化
    new_factor_std_df = pd.DataFrame(stats.zscore(new_factor_df, nan_policy='omit',axis=1), columns=new_factor_df.columns,
                                      index=new_factor_df.index)
    print("再做一次Z-score标准化后得到的因子值表：")
    print(new_factor_std_df)


    # 对股票原始价格进行后复权处理，用后复权价计算日收益率
    AF = DayPrice_stock['AF']
    reright_price = stock(DayPrice_stock) * AF
    # print("复权后价格：")
    # print(reright_price)
    stock_return = reright_price.pct_change(periods=1)  # 对应的日收益率表（复权价）
    stock_return = stock_return.loc[factor.index]
    stock_return = stock_return.fillna(0)
    print("复权价计算的股票池日收益率：")
    print(stock_return)


    # 因子评价体系开始
    # 1. 回归法
    factor_return = []    # 因子收益率序列
    t_values = []         # t值序列
    for n in new_factor_std_df.index:
        x = new_factor_std_df.loc[n].dropna()
        y = stock_return.loc[n].dropna()
        index = x.index & y.index
        x = x.loc[index]
        y = y.loc[index]
        x = sm.add_constant(x)
        est = sm.OLS(y,x)
        model = est.fit()
        factor_return.append(model.params[n])
        t_values.append(model.tvalues[n])

    print("1. 回归法")
    a = larger_than_prob(pd.Series(factor_return),0)
    b = pd.Series(factor_return).mean()
    c = pd.Series(t_values).mean()
    d = larger_than_prob(abs(pd.Series(t_values)),0)
    e = larger_than_prob(abs(pd.Series(t_values)),2)
    print("因子收益率大于0的概率："+str(a))
    print("因子收益率均值："+str(b))
    print("t值绝对值的均值："+str(c))
    print("t值绝对值大于0的概率："+str(d))
    print("t值绝对值大于2的概率：" + str(e))


    # 2. IC体系：初步检验
    # normal IC
    normal_IC = []    # normal IC 序列
    count = -1
    for n in new_factor_std_df.index:
        count = count + 1                      #目前的位置
        if count != len(new_factor_std_df)-1:
            # x = new_factor_std_df.loc[n].dropna()
            x = new_factor_df.loc[n].dropna()
            y = stock_return.iloc[count+1].dropna()
            index = x.index & y.index
            x = x.loc[index]
            y = y.loc[index]
            df = pd.DataFrame({'factor': x, 'return': y})
            corr_array = df.corr(method='pearson')
            corr = corr_array.loc['factor']['return']
            normal_IC.append(corr)
        else:
            break

    print("2. IC检验体系：")
    normal_IC = pd.Series(normal_IC)
    normal_IC_mean = normal_IC.mean()
    normal_IC_std = normal_IC.std()
    normal_IC_IR = normal_IC_mean/normal_IC_std
    a = larger_than_prob(normal_IC,0)
    b = larger_than_prob(abs(normal_IC),0.02)
    print("IC均值为："+str(normal_IC_mean))
    print("IC标准差为："+str(normal_IC_std))
    print("IC>0的比例："+str(a))
    print("IC绝对值大于0.02的比例："+str(b))
    print("IR为："+str(normal_IC_IR))


    # 3. 分组回溯法
    print("3. 分组回溯法")
    # 挑出月调仓日
    tempt = new_factor_std_df.resample('M').sum()
    tempt = tempt.reset_index()
    tempt.index = tempt['date'].apply(lambda x: x.strftime('%Y-%m'))
    monthly_trade_date = sel_monthly_trade_date(new_factor_std_df,tempt.index)
    # print("所有月调仓日如下：")
    # print(monthly_trade_date)
    print("回溯期内共交易"+str(len(monthly_trade_date))+"次")

    # 存储183次调仓 每次调仓的分组结果
    portfolio = {}
    for n in monthly_trade_date:
        n = n.strftime("%Y-%m-%d")
        x = div_group(new_factor_std_df.loc[n])
        portfolio[n] = x

    """
    成功获得在183个月调仓日上根据因子值大小划分的五个组的股票组合
    存在portfolio字典内 可通过日期调用
    group1因子值最小 group5因子值最大
    """

    # 计算五个组每日收益率
    new_dict = {}
    daily_date = stock_return.index
    for n in daily_date[0:100]:    # 先用调仓日当天测试
        n_firstday = get_first_day(stock_return, n)
        n = n.strftime("%Y-%m-%d")
        list = []
        for s in ['group1','group2','group3','group4','group5']:
            group = portfolio[n_firstday][s]
            group_return = 0
            weight = 1/len(group)
            NaN_stock_return_list = []
            for x in group:
                group_return = group_return + weight * stock_return.loc[n][x]  # 将NaN替换成0 跳过if判断条件
            list.append(group_return)
        new_dict[n] = list


    # 跑数据的时候用
    date_return_df = pd.DataFrame(new_dict,index=['group1','group2','group3','group4','group5'])
    date_return_df.columns.name = 'date'
    date_return_df = date_return_df.T
    date_return_df['top_bottom'] = (date_return_df['group1'] - date_return_df['group5']) / 2
    print(date_return_df)
    # date_return_df.to_csv("ep_date_return.csv")
    date_return_df.index = pd.to_datetime(date_return_df.index)

    # date_return_df.index.name = 'date'
    # date_return_df = date_return_df.reindex()
    # print(date_return_df)
    # print(date_return_df.index.dtype)


    # 直接读取已经跑好的数据
    # date_return_df = pd.read_csv("date_return_df_reright.csv")
    # date_return_df.index = pd.to_datetime(date_return_df['date'])
    # date_return_df = date_return_df.drop('date', axis=1)


    print("各组日收益率表如下：")
    print(date_return_df)
    print(date_return_df.index.dtype)
    new_date_return_df = date_return_df + 1

    # 计算净值变化
    net_value_df = new_date_return_df.copy(deep=True)
    net_value_df['net_value_g1'] = net_value_df['group1'].cumprod(axis=0)
    net_value_df['net_value_g2'] = net_value_df['group2'].cumprod(axis=0)
    net_value_df['net_value_g3'] = net_value_df['group3'].cumprod(axis=0)
    net_value_df['net_value_g4'] = net_value_df['group4'].cumprod(axis=0)
    net_value_df['net_value_g5'] = net_value_df['group5'].cumprod(axis=0)
    net_value_df['net_value_tb'] = net_value_df['top_bottom'].cumprod(axis=0)
    # net_value_df['relative_intensity'] = net_value_df['net_value_g1']/net_value_df['net_value_g5']
    # net_value_df.to_csv("ep_net_value.csv")
    net_value_df = net_value_df.drop(['group1','group2','group3','group4','group5','top_bottom'],axis=1)
    print("各组净值变化情况如下：")
    print(net_value_df)
    net_value_df.plot()           # 绘制净值曲线图
    plt.show()

    length = len(date_return_df)
    annul_return_am_g1 = annul_return_am(net_value_df.iloc[-1]['net_value_g1'],length)
    annul_return_am_g2 = annul_return_am(net_value_df.iloc[-1]['net_value_g2'], length)
    annul_return_am_g3 = annul_return_am(net_value_df.iloc[-1]['net_value_g3'], length)
    annul_return_am_g4 = annul_return_am(net_value_df.iloc[-1]['net_value_g4'], length)
    annul_return_am_g5 = annul_return_am(net_value_df.iloc[-1]['net_value_g5'], length)
    annul_return_am_tb = annul_return_am(net_value_df.iloc[-1]['net_value_tb'], length)
    annul_return_am_list = [annul_return_am_g1,annul_return_am_g2,annul_return_am_g3,annul_return_am_g4,annul_return_am_g5,annul_return_am_tb]
    print("各组的平均年化收益率为：")
    print(annul_return_am_list)

    # 计算最大回撤
    m_drawback_g1 = max_drawback(net_value_df['net_value_g1'])
    m_drawback_g2 = max_drawback(net_value_df['net_value_g2'])
    m_drawback_g3 = max_drawback(net_value_df['net_value_g3'])
    m_drawback_g4 = max_drawback(net_value_df['net_value_g4'])
    m_drawback_g5 = max_drawback(net_value_df['net_value_g5'])
    m_drawback_tb = max_drawback(net_value_df['net_value_tb'])
    m_drawback_list = [m_drawback_g1,m_drawback_g2,m_drawback_g3,m_drawback_g4,m_drawback_g5,m_drawback_tb]
    print("各组的最大回撤率为：")
    print(m_drawback_list)

    # 计算夏普比
    sharpe_ratio_g1 = sharpe_ratio(date_return_df['group1'],net_value_df.iloc[-1]['net_value_g1'])
    sharpe_ratio_g2 = sharpe_ratio(date_return_df['group2'],net_value_df.iloc[-1]['net_value_g2'])
    sharpe_ratio_g3 = sharpe_ratio(date_return_df['group3'],net_value_df.iloc[-1]['net_value_g3'])
    sharpe_ratio_g4 = sharpe_ratio(date_return_df['group4'],net_value_df.iloc[-1]['net_value_g4'])
    sharpe_ratio_g5 = sharpe_ratio(date_return_df['group5'],net_value_df.iloc[-1]['net_value_g5'])
    sharpe_ratio_tb = sharpe_ratio(date_return_df['top_bottom'], net_value_df.iloc[-1]['net_value_tb'])
    sharpe_ratio_list = [sharpe_ratio_g1,sharpe_ratio_g2,sharpe_ratio_g3,sharpe_ratio_g4,sharpe_ratio_g5,sharpe_ratio_tb]
    print("各组的夏普比为：")
    print(sharpe_ratio_list)



    # time_end = time.time()
    # print('程序运行共需' + str(end - start) + '秒')







