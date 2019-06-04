import yfinance as yf
import csv
import numpy as np
import pandas
import statistics
import quandl
import os
import datetime
import random
import pandas as pd
import matplotlib.pyplot as plt

from git import Repo

# rorepo is a Repo instance pointing to the git-python repository.
# For all you know, the first argument to Repo is a path to the repository
# you want to work with
repo_path = os.getenv('GIT_REPO_PATH')
repo = Repo(repo_path)


now = datetime.date.today()
date = str(now)


outputFile = []

spy = yf.Ticker("SPY")
marketHist = spy.history(period="1y")
marketCloses = marketHist['Close'].pct_change().tolist()
marketCloses = marketCloses[1:]

reader = csv.reader(open("etfs.csv", "r"), delimiter=",")
etfs = list(reader)[0]
print(etfs)
returns = []
prices = []
tickers = []
i = 0
etfs=etfs[0:10]
for etf in etfs:
    print(i)
    i += 1
    etf = etf.replace(u'\xa0', u'')
    etf = etf.replace(u'\ufeff', u'')
    tickers.append(etf)
    company = yf.Ticker(etf)
    hist = company.history(period="2y")
    closes = hist['Close'].pct_change().tolist()
    closes = closes[1:]
    if(len(closes) < 503):
        closes = closes[1:]
        for k in range(0, 503-len(closes)):
            closes.append(0)
    print(len(closes))
    prices.append(closes)


returns_daily = np.array([np.array(xi) for xi in prices])
returns_annual = []
for comp in returns_daily:
    print(comp)
    print(comp.mean())
    returns_annual.append(comp.mean()*250)
print(returns_annual)
cov_daily = []
for j in range(0, len(returns_daily)):
    temp = []
    for k in range(0, len(returns_daily)):
        cov = np.cov(returns_daily[j], returns_daily[k])
        temp.append(cov)
    cov_daily.append(temp)
cov_daily = np.cov(returns_daily)
print(cov_daily)
cov_annual = cov_daily * 250

variances = []
for j in range(0, len(cov_daily)):
    variances.append(cov_annual[j][j])
prices=np.transpose([np.append("Ticker", tickers), np.append("2-Year Average Annual Returns", returns_annual), np.append("Variance of Daily Returns" ,variances)])
port_returns = []
port_volatility = []
stock_weights = []
sharpe_ratio = []
# set the number of combinations for imaginary portfolios
num_assets = len(etfs)
num_portfolios = 50000

# populate the empty lists with each portfolios returns,risk and weights
for single_portfolio in range(num_portfolios):
    weights = np.random.random(num_assets)
    weights /= np.sum(weights)
    returns = np.dot(weights, returns_annual)
    volatility = np.sqrt(np.dot(weights.T, np.dot(cov_annual, weights)))
    port_returns.append(returns)
    sharpe = returns / volatility
    sharpe_ratio.append(sharpe)
    port_volatility.append(volatility)
    stock_weights.append(weights)

# a dictionary for Returns and Risk values of each portfolio
portfolio = {'Returns': port_returns,
             'Volatility': port_volatility,
             'Sharpe Ratio': sharpe_ratio}

# extend original dictionary to accomodate each ticker and weight in the portfolio
for counter,symbol in enumerate(etfs):
    portfolio[symbol+' Weight'] = [Weight[counter] for Weight in stock_weights]

# make a nice dataframe of the extended dictionary
df = pd.DataFrame(portfolio)

# get better labels for desired arrangement of columns
column_order = ['Returns', 'Volatility', 'Sharpe Ratio'] + [stock+' Weight' for stock in etfs]

# reorder dataframe columns
df = df[column_order]

# plot the efficient frontier with a scatter plot
min_volatility = df['Volatility'].min()
max_sharpe = df['Sharpe Ratio'].max()

# use the min, max values to locate and create the two special portfolios
sharpe_portfolio = df.loc[df['Sharpe Ratio'] == max_sharpe]
min_variance_port = df.loc[df['Volatility'] == min_volatility]

# plot frontier, max sharpe & min Volatility values with a scatterplot
plt.style.use('seaborn-dark')
df.plot.scatter(x='Volatility', y='Returns', c='Sharpe Ratio',
                cmap='RdYlGn', edgecolors='black', figsize=(10, 8), grid=True)
plt.scatter(x=sharpe_portfolio['Volatility'], y=sharpe_portfolio['Returns'], c='red', marker='D', s=200)
plt.scatter(x=min_variance_port['Volatility'], y=min_variance_port['Returns'], c='blue', marker='D', s=200 )
plt.xlabel('Volatility (Std. Deviation)')
plt.ylabel('Expected Returns')
plt.title('Efficient Frontier: ' + date)
plt.savefig('efficient-frontier.png')
plt.show()



with open('etf-data.csv', "w", newline='') as f:
    writer = csv.writer(f)
    writer.writerows(prices)   

with open('covariance-matrix.csv', "w", newline='') as f:
    writer = csv.writer(f)
    writer.writerows(cov_daily) 

file_list = [
    'efficient-frontier.png',
    'etf-data.csv',
    'covariance-matrix.csv'
]
commit_message = 'Add graph'
repo.index.add(file_list)
repo.index.commit(commit_message)
origin = repo.remote('origin')
origin.push()

print(min_variance_port.T)
print(sharpe_portfolio.T)
 



