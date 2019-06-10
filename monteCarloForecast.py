import yfinance as yf
from yahoofinancials import YahooFinancials
import csv
import numpy as np
import pandas
import statistics
import os
import datetime
import random
import pandas as pd
import matplotlib.pyplot as plt

import progressbar

import scipy.optimize
import random
from numpy import matrix, array, zeros, empty, sqrt, ones, dot, append, mean, cov, transpose, linspace
from numpy.linalg import inv, pinv

from git import Repo

initial_amount = input("Intial Portfolio Balance: ")
if(initial_amount[0:1] == "$"):
        initial_amount = float(initial_amount[1:])
else:
        initial_amount = float(initial_amount)

withdrawal_amount = input("Annual Withdrawal Amount: ")
if(withdrawal_amount == ""):
        withdrawal_amount = 0
if(withdrawal_amount[0:1] == "$"):
        withdrawal_amount = float(withdrawal_amount[1:])
else:
        withdrawal_amount = float(withdrawal_amount)

contribution_amount = input("Annual Contribution Amount: ")
if(contribution_amount == ""):
        contribution_amount = 0
if(contribution_amount[0:1] == "$"):
        contribution_amount = float(contribution_amount[1:])
else:
        contribution_amount = float(contribution_amount)

inflation_rate = input("Annual Inflation Rate (Default: .02): ")
if(inflation_rate == ""):
        inflation_rate = .02
inflation_rate = float(inflation_rate)

simulation_period = int(input("Simulation Period (years): "))

historical = False
time_period = "10y"


risk_free_rate = input("Annual Risk Free Rate (Default is .02): ")
try:
        risk_free_rate = float(risk_free_rate)
        if(risk_free_rate >= 1):
                risk_free_rate /= 100
except:
        risk_free_rate = .02
risk_free_rate = risk_free_rate/12


risk_premium_by_class = input("Would you like to vary risk premium by asset class? (y/n) ")
set_risk_manually = input("Would you like to set risk premia manually? (y/n) ")
if (risk_premium_by_class == "y" or risk_premium_by_class == "Y" or risk_premium_by_class == "yes" or risk_premium_by_class == "Yes"):
        risk_premium_by_class = True
        if (set_risk_manually == "y" or set_risk_manually == "Y" or set_risk_manually == "yes" or set_risk_manually == "Yes"):
                try:
                        risk_premium_equity = float(input("Equity Risk Premium (Default: .05): "))/12
                        risk_premium_fi = float(input("Fixed Income Risk Premium (Default: .02): "))/12
                        risk_premium_commodity = float(input("Commodity Risk Premium (Default: .01): "))/12
                except:
                        risk_premium_equity = .05/12
                        risk_premium_fi = .02/12
                        risk_premium_commodity = .01/12

        else:
                risk_premium_equity = .05/12
                risk_premium_fi = .02/12
                risk_premium_commodity = .01/12
else:
        if (set_risk_manually == "y" or set_risk_manually == "Y" or set_risk_manually == "yes" or set_risk_manually == "Yes"):
                try:
                        risk_premium = float(input("Market Risk Premium (Default: .05): "))/12
                except: 
                        risk_premium = .05/12
        else:
                risk_premium = .05/12


edit_covariances = input("Would you like to manually edit covariances? (y/n): ")
if(edit_covariances == "y" or edit_covariances == "Y" or edit_covariances == "Yes" or edit_covariances == "yes"):
        edit_covariances = True
else:
        edit_covariances = False





def assets_meanvar(names, prices, caps):
        prices = matrix(prices)                         # create numpy matrix from prices
        weights = array(caps) / sum(caps)       # create weights

        # create matrix of historical returns
        rows, cols = prices.shape
        returns = empty([rows, cols-1])
        for r in range(rows):
                for c in range(cols-1):
                        p0, p1 = prices[r,c], prices[r,c+1]
                        returns[r,c] = (p1/p0)-1

        # calculate expected returns
        expreturns = array([])
        for r in range(rows):
                expreturns = append(expreturns, mean(returns[r]))
        # calculate covariances
        covars = cov(returns)

        expreturns = (1+expreturns)**250-1      # Annualize expected returns
        covars = covars * 250                           # Annualize covariances

        return names, weights, expreturns, covars


# Calculates portfolio mean return
def port_mean(W, R):
        return sum(R*W)

# Calculates portfolio variance of returns
def port_var(W, C):
        return dot(dot(W, C), W)





# Combination of the two functions above - mean and variance of returns calculation
def port_mean_var(W, R, C):
        return port_mean(W, R), port_var(W, C)


now = datetime.date.today()
date = str(now)


outputFile = []

print("Pulling Market Data...")
spy = yf.Ticker("SPY")
marketHist = spy.history(period=time_period,interval="1d")
marketCloses = marketHist['Close'].tolist()
fullLength = len(marketCloses)
temp = []
for j in range(0, len(marketCloses)):
    if(j%21 == 0):
        temp.append(marketCloses[j])
marketCloses=temp
marketReturns = pd.Series(marketCloses).pct_change().tolist()[1:]

if(risk_premium_by_class == True):
        agg = yf.Ticker("AGG")
        fiHist = agg.history(period=time_period,interval="1d")
        fiCloses = fiHist['Close'].tolist()
        fullLength = len(fiCloses)
        temp = []
        for j in range(0, len(fiCloses)):
                if(j%21 == 0):
                        temp.append(fiCloses[j])
        fiCloses=temp
        fiReturns = pd.Series(fiCloses).pct_change().tolist()[1:]

        gld = yf.Ticker("GLD")
        commHist = gld.history(period=time_period,interval="1d")
        commCloses = commHist['Close'].tolist()
        fullLength = len(commCloses)
        temp = []
        for j in range(0, len(commCloses)):
                if(j%21 == 0):
                        temp.append(commCloses[j])
        commCloses=temp
        commodityReturns = pd.Series(commCloses).pct_change().tolist()[1:]


print("Calculating Market Risk Aversion...")
if(risk_premium_by_class == True):
        equityExcessReturns = [x-risk_free_rate for x in marketReturns]
        equityExcessReturnsVar = np.var(equityExcessReturns)
        # fiExcessReturns = [x-risk_free_rate for x in fiReturns]
        # fiExcessReturnsVar = np.var(fiExcessReturns)
        # commodityExcessReturns = [x-risk_free_rate for x in commodityReturns]
        # commodityExcessReturnsVar = np.var(commodityExcessReturns)

        risk_aversion_equity = risk_premium_equity/equityExcessReturnsVar
        risk_aversion_fi = risk_premium_fi/equityExcessReturnsVar
        risk_aversion_commodity = risk_premium_commodity/equityExcessReturnsVar
else:
        risk_premium = ((sum(marketReturns)/len(marketReturns))-risk_free_rate)
        marketExcessReturns = [x-risk_free_rate for x in marketReturns]
        marketExcessReturnsVar = np.var(marketExcessReturns)
        risk_aversion_equity = risk_premium/marketExcessReturnsVar
        risk_aversion_fi = risk_premium/marketExcessReturnsVar
        risk_aversion_commodity = risk_premium/marketExcessReturnsVar



reader = csv.reader(open("etfAUM.csv", "r",encoding="utf8", errors='ignore'), delimiter=",")
etfs = list(reader)
assetClass = list(np.transpose(etfs)[3])
aum = list(np.transpose(etfs)[2])
etfs = list(np.transpose(etfs)[0])
etfNames = list(np.transpose(etfs)[1])
temp = []


print("Pulling ETF Data...")
bar = progressbar.ProgressBar(maxval=100, widgets=[progressbar.Bar('=', '[', ']'), ' ', progressbar.Percentage()])
bar.start()
for i in range(0, len(aum)):
        temp.append(float(aum[i]))
aum = temp

returns = []
prices = []
tickers = []
marketCaps = []
i = 0
for etf in etfs:
    bar.update(i+1)
    i += 1
    etf = etf.replace(u'\xa0', u'')
    etf = etf.replace(u'\ufeff', u'')
    company = yf.Ticker(etf)
    hist = company.history(period=time_period, interval="1d")
    closes = hist['Close'].tolist()

    if(len(closes) < fullLength):
        continue;

    marketCaps.append(aum[i-1])
    tickers.append(etf)

    temp = []
    for j in range(0, len(closes)):
        if(j%21 == 0):
            temp.append(closes[j])
    closes=pd.Series(temp).pct_change().tolist()[1:]
    prices.append(closes)
bar.finish()

assetWeights = np.transpose([tickers, np.zeros(len(tickers))])
with open('asset-weights.csv', "w", newline='') as f:
                writer = csv.writer(f)
                writer.writerows(assetWeights)
print("")
print("")
input("Inside your current folder, open the file titled 'asset-weights.csv' to adjust the weighting of each ETF in the simulated portfolio. Make sure that all of the weights add up to 1. When you have finished, save the file, return to this window, and press enter ")
print("")
reader = csv.reader(open("asset-weights.csv", "r",encoding="utf8", errors='ignore'), delimiter=",")
weights = list(reader)
weights = list(np.transpose(weights)[1])
temp = []
for i in range(0, len(weights)):
        temp.append(float(weights[i]))
W = np.asarray(temp)

print("Computing returns and covariances...")
temp = []
marketCapSum = sum(marketCaps)
for marketCap in marketCaps:
        temp.append(marketCap/marketCapSum)
marketCapWeights = temp




returns_monthly = np.array([np.array(xi) for xi in prices])
returns_annual = []
for comp in returns_monthly:
    returns_annual.append(comp.mean()*12)
cov_monthly = []
cov_monthly = np.cov(returns_monthly)
cov_annual = cov_monthly * 12
if(edit_covariances == True):
        covariance = [[" "]]
        for i in range(0, len(cov_annual)):
                covariance[0].append(tickers[i])
        for j in range(0, len(cov_annual)):
                temp = []
                temp.append(tickers[j])
                for k in range(0, len(cov_annual)):
                        temp.append(cov_annual[j][k])
                covariance.append(temp)

        with open('covariance-matrix.csv', "w", newline='') as f:
                writer = csv.writer(f)
                writer.writerows(covariance)
        print("")
        print("")
        input("The historical annual covariance matrix has been saved as a file called 'covariance-matrix.csv'. Open the file and manually adjust any covariances you would like to change, then save the file, return to this window, and press enter to continue ")
        print("")
        reader = csv.reader(open("covariance-matrix.csv", "r",encoding="utf8", errors='ignore'), delimiter=",")
        cov_annual = list(reader)
        cov_annual = cov_annual[1:]
        cov_annual = np.transpose(np.transpose(cov_annual)[1:])
        temp = []
        for i in range(0, len(cov_annual)):
                temp.append([])
                for j in range(0, len(cov_annual)):
                        temp[i].append(float(cov_annual[i][j]))
        cov_annual = np.asarray(temp)


temp = []
for i in range(0, len(returns_monthly)):
    market_cov = np.cov(returns_monthly[i], marketReturns)[0][1]*12
    asset_var = np.cov(returns_monthly[i], returns_monthly[i])[1][1]*12
    temp.append(market_cov/asset_var)
beta_annual = temp

print("Computing Implied Excess Returns")
cov_scaled = []
for i in range(0, len(cov_annual)):
        cov_scaled.append([])
        for j in range(0, len(cov_annual[i])):
                if(assetClass[i] == "Equity"):
                        cov_scaled[i].append(cov_annual[i][j] * risk_aversion_equity)
                elif(assetClass[i] == "Fixed Income"):
                        cov_scaled[i].append(cov_annual[i][j] * risk_aversion_fi)
                else:
                        cov_scaled[i].append(cov_annual[i][j] * risk_aversion_commodity)
excess = np.matmul(cov_scaled,marketCapWeights)
returns_annual = excess

variances = []
for j in range(0, len(cov_monthly)):
    variances.append(cov_annual[j][j])
prices=np.transpose([np.append("Ticker", tickers), np.append("2-Year Average Annual Returns", returns_annual), np.append("Variance of Daily Returns" ,variances)])

port_mean, port_var = port_mean_var(W, returns_annual, cov_annual)
print("Portfolio Mean Annual Returns: ", port_mean)
print("Portfolio Standard Deviation of Returns: ", port_var**.5)

num_simulations = 100000
ending_amounts = []
for i in range(0, num_simulations):
        balance = initial_amount
        for j in range(0, simulation_period):
                annual_return = np.random.normal(port_mean, port_var**.5)
                balance = balance * (1+annual_return)
                balance = balance - withdrawal_amount + contribution_amount
                balance = balance * (1-inflation_rate)
        ending_amounts.append(balance)
print(sum(ending_amounts)/len(ending_amounts))

plt.hist(ending_amounts, bins=100)
plt.title("Monte Carlo Simulation Results")
plt.xlabel("Ending Balance (Inflation-Adjusted $)")
plt.ylabel("Number of Simulated Outcomes")
plt.savefig("monte-carlo.png")
plt.show()
