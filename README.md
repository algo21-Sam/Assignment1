# Assignment1
## 1. 摘要：
Factor investing 因子投资-单因子测试框架：

本项目主要参考以下两篇研报构建了单因子测试框架，采用了**回归法、IC评价体系以及分组回溯法**来对单因子选股效果进行评价

## 2. 参考研报：
20170410-光大证券-光大证券多因子系列报告之一：因子测试框架

20180306-方正证券-方正证券“远山”量化选股系列（一）：规矩，方正单因子测试之评价体系

## 3. 代码逻辑及说明：
（1）第一步：数据读取/输入

这里所指的数据包括**股票行情数据、因子值以及行业分类（后用于行业中性化）**，其中本项目中股票行情数据、因子值和行业分类数据均为.mat格式。其中因子值的读取也提供了其他两种方式：通过.csv读取或通过具体计算方式获取

（2）第二步：数据预处理

包括一系列对数据类型以及格式的处理，如：将array序列转为list、将array序列转为datetime、将date_num转化为list、统一日期：日期对齐、获取股票池收盘价数据等

（3）第三步：因子值加工

异常值处理：MAD中位数去极值法  --> z-score标准化/归一化  --> **行业、市值中性化**  --> 再次z-score标准化/归一化

（4）第四步：收益率加工

在原始日收益率基础上，乘上复权因子计算**复权日收益率**

（5）第五步：因子评价体系

回归法：计算因子收益率序列及t值序列，进而输出因子收益率大于0的概率、因子收益率均值、t值绝对值的均值、t值绝对值大于0的概率、t值绝对值大于2的概率等指标

IC体系：在每个时间截面上计算IC值得到normal IC序列，进而输出**IC均值**、IC标准差、**IC>0的比例**、**IC的绝对值>=0.02的比例**、IR等指标

**分组回溯法**：月末调仓，根据因子值大小将股票池分为5组+多空组，计算五个组的日收益率从而算出每组的净值变化序列，最终输出净值曲线以及各种回测指标（年化收益率、最大回撤、夏普比等）




## 4. 输出结果：
对MV市值因子进行测试，结果如下：

（1）回归法：

因子收益率大于0的概率：0.4896

因子收益率均值：0.0003

t值绝对值的均值：0.3170

t值绝对值大于0的概率：1.0000

t值绝对值大于2的概率：0.5957

（2）IC体系

IC均值：-0.0153

IC标准差：0.0887

IC>0的比例：0.3889

IC绝对值大于0.02的比例：0.8272

IR：-0.1725

（3）分组回溯法（注：回溯期内共交易183次，group1为因子值最小的，group5为因子值最大，top_bottom多空组买多group1卖空group5）

各组净值曲线图：

![净值曲线](https://github.com/algo21-Sam/Assignment1/blob/master/net_value.jpg)

各组年化收益率：

Group1：0.4294

Group2：0.1960

Group3：0.1016

Group4：-0.0011

Group5：0.1797

Top-bottom：0.3179

各组的最大回撤率：

Group1：0.6492

Group2：0.6824

Group3：0.7189

Group4：0.7742

Group5：0.9780

Top-bottom：0.0412

各组的夏普比：

Group1：1.4684

Group2：0.6531

Group3：0.3369

Group4：-0.0039

Group5：-0.6063

Top-bottom：7.3206








