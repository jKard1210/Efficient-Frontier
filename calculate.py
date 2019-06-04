import yfinance as yf
import csv
import numpy as np
import pandas
import statistics
import quandl
import random
import pandas as pd
import matplotlib.pyplot as plt


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
for etf in etfs:
    print(i)
    i += 1
    temp = []
    etf = etf.replace(u'\xa0', u'')
    etf = etf.replace(u'\ufeff', u'')
    company = yf.Ticker(etf)
    try:
        hist = company.history(period="2y")
        closes = hist['Close'].pct_change().tolist()
        closes = closes[1:]
        if(len(closes) < 503):
            closes = closes[1:]
            for k in range(0, 503-len(closes)):
                closes.append(0)
        print(len(closes))
        prices.append(closes)


    except:
        temp.append(0)
        temp.append(0)
        temp.append(0)
        temp.append(0)

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
plt.title('Efficient Frontier')
plt.show()

print(min_variance_port.T)
print(sharpe_portfolio.T)

with open("results.csv", "w", newline='') as f:
    writer = csv.writer(f)
    writer.writerows(prices)    



