# Assignment1
## 1. Abstract：
Factor investing -Single Factor Testing Framework：

This project mainly refers to the following two research reports to construct a single factor testing framework, and uses **regression method, IC evaluation system and grouping backtracking method** to evaluate the effect of single factor stock selection

## 2. Research paper：
20170410-光大证券-光大证券多因子系列报告之一：因子测试框架

20180306-方正证券-方正证券“远山”量化选股系列（一）：规矩，方正单因子测试之评价体系

## 3. Code logic and explaination：
（1）Step1：input data

The data here include **stock market data, factor value and industry classification (later used for industry neutrality)**, where the stock market data, factor value and industry classification data used in this project are in .mat format. There are also two other ways to read the factor value: reading csv file or through concrete calculation.

（2）Step2：data preprocessing

Includes a series of data type and format processing, such as: Array sequence to list, Array sequence to datetime, date_num to list, unified date: date alignment, obtaining stock pool closing price data, etc

（3）Step3：factor value processing

Outlier treatment: MAD median de-extremalization method --> Z-score standardization/normalization --> **industry, market capitalization neutrality** -->restandardization/normalization again

（4）Step4：return rate processing 

On the basis of the original daily rate of return, multiply by the compound weighting factor to calculate **daily rate of return**

（5）Step5：factor evaluation system

Regression method: calculate the sequence of factor rate of return and T-value sequence, and then output the probability of factor rate of return greater than 0, the mean value of factor rate of return, the mean value of absolute value of T-value, the probability of absolute value of T-value greater than 0, the probability of absolute value of T-value greater than 2 and other indicators

IC system: Calculate IC value to Normal IC sequence on each time section, and then output **IC mean**, IC standard deviation, **proportion of IC>0,** **proportion of IC absolute value >0.02,** IR and other indexes

**Group backtracking method** : adjust the position at the end of the month, divide the stock pool into 5 groups plus a "long-short group" according to the value of the factor, calculate the daily return rate of the five groups so as to calculate the sequence of net value change of each group, and finally output the net value curve and various backtest indicators (annualized return rate, maximum retraction, Sharpe ratio, etc.)

## 4. Output result：
Testing the market value factor, here's the result:

（1）Regressiong Method：

Probability of factor return greater than 0:0.4896

Average factor rate of return: 0.0003

The mean value of the absolute value of T-values: 0.3170

Probability of T-value greater than 0: 1.0000

Probability of T-value greater than 2: 0.5957


（2）IC System

IC mean value: -0.0153

IC standard deviation: 0.0887

Ratio of IC>0:0.3889

The proportion of IC with absolute value greater than 0.02: 0.8272

（3）Grouping backtracking method 
(Note: there were 183 transactions in the backtracking period, Group1 was the one with the smallest factor value, Group5 was the one with the largest factor value, Top_bottom group bought long Group1 and shorted Group5)

netvalue curve of each group:

![netvalue curve](https://github.com/algo21-Sam/Assignment1/blob/master/net_value.jpg)

Annulized return rate of each group:

Group1：0.4294

Group2：0.1960

Group3：0.1016

Group4：-0.0011

Group5：0.1797

Top-bottom：0.3179

Maximum-drawdown of each group:

Group1：0.6492

Group2：0.6824

Group3：0.7189

Group4：0.7742

Group5：0.9780

Top-bottom：0.0412

Sharpe Ratio of each group

Group1：1.4684

Group2：0.6531

Group3：0.3369

Group4：-0.0039

Group5：-0.6063

Top-bottom：7.3206








