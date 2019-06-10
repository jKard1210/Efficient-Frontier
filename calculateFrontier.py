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



historical = input("Would you like to create a historical or forecasted frontier? (h/f): ")
if(historical == "h" or historical=="H" or historical == "Historical" or historical=="historical"):
        historical = True
else:
        historical = False

#Time Period Options: 1y, 5y, 10y
set_period_manually = False
if(historical ==  False):
        time_period = "10y"
else:
        time_period = input("Choose Time Period (1y, 5y, 10y, Custom): ")
        if(time_period == "1"):
                time_period = "1y"
        elif(time_period == "5"):
                time_period = "5y"
        elif(time_period == "10"):
                time_period = "10y"
        elif(time_period == "custom" or time_period == "Custom"):
                set_period_manually = True
                start_date = input("Start Date (yyyy-mm-dd):")
                end_date = input("End Date (yyyy-mm-dd):")
        elif(time_period == "" or (time_period != "1y" and time_period != "5y" and time_period != "10y")):
                time_period = "10y"

#Risk Free Rate
if(historical == False):
        risk_free_rate = input("Annual Risk Free Rate (Default is .02): ")
        try:
                risk_free_rate = float(risk_free_rate)
                if(risk_free_rate >= 1):
                        risk_free_rate /= 100
        except:
                risk_free_rate = .02
else:
        risk_free_rate = .02
risk_free_rate = risk_free_rate/12


#X-Variable Options: Volatility, Beta
x_variable = input("X-Variable (Volatility, Beta): ")
if(x_variable == "V" or x_variable == "v" or x_variable == "volatility"):
        x_variable = "Volatility"
elif(x_variable == "B" or x_variable == "b" or x_variable == "beta"):
        x_variable = "Beta"
elif(x_variable != "Volatility" and x_variable!= "Beta"):
        x_variable = "Volatility"

#Y-Variable Options:  Sharpe Ratio, Treynor Ratio, Implied Excess Return
if(historical == False):
        y_variable = input("Y-Variable (Implied Excess Return, Sharpe Ratio, Treynor Ratio): ")
        if(y_variable == "I" or y_variable == "i" or y_variable == "implied excess return"):
                y_variable = "Implied Excess Return"
        elif(y_variable == "S" or y_variable == "s" or y_variable == "sharpe ratio"):
                y_variable = "Sharpe Ratio"
        elif(y_variable == "T" or y_variable == "t" or y_variable == "treynor ratio"):
                y_variable = "Treynor Ratio"
        elif(y_variable != "Implied Excess Return" and y_variable!= "Sharpe Ratio" and y_variable!= "Treynor Ratio"):
                y_variable = "Implied Excess Return"
else:
        y_variable = input("Y-Variable (Real Return, Nominal Return, Sharpe Ratio, Treynor Ratio): ")
        if(y_variable == "N" or y_variable == "n" or y_variable == "nominal return"):
                y_variable = "Nominal Return"
        elif(y_variable == "S" or y_variable == "s" or y_variable == "sharpe ratio"):
                y_variable = "Sharpe Ratio"
        elif(y_variable == "R" or y_variable == "r" or y_variable == "real return"):
                y_variable = "Real Return"
        elif(y_variable == "T" or y_variable == "t" or y_variable == "treynor ratio"):
                y_variable = "Treynor Ratio"
        elif(y_variable != "Nominal Return" and y_variable != "Real Return" and y_variable!="Sharpe Ratio" and y_variable!= "Treynor Ratio"):
                y_variable = "Nominal Return"

#Gradient-Variable Options: None, Volatility, Beta, Sharpe Ratio, Treynor Ratio
gradient_variable = input("Gradient Variable (None, Sharpe Ratio, Treynor Ratio, Beta): ")
if(gradient_variable == "s" or gradient_variable == "S"):
        gradient_variable = "Sharpe Ratio"
elif(gradient_variable == "t" or gradient_variable == "T"):
        gradient_variable = "Treynor Ratio"
elif(gradient_variable == "b" or gradient_variable == "B"):
        gradient_variable = "Beta"
elif(gradient_variable != "Sharpe Ratio" and gradient_variable!= "Treynor Ratio" and gradient_variable!= "Beta"):
        gradient_variable = "None"


#Inflation Factor
inflation_discount = .02
if(y_variable == "Real Return"):
        inflation_discount = input("Inflation Rate (Default: .02): ")
inflation_discount = .02
inflation_factor = 1/(1+inflation_discount)

#Risk-Aversion Coefficient


risk_premium = .02
risk_premium = risk_premium/12
risk_premium_by_class = False
if(historical == False):
        risk_premium_by_class = input("Would you like to vary risk premium by asset class? (y/n) ")
        set_risk_manually = input("Would you like to set risk premium manually? (y/n) ")
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


graph_all = input("Would you like to include individual labelled ETFs on graph? (y/n) ")
if (graph_all == "y" or graph_all == "Y" or graph_all == "yes" or graph_all == "Yes"):
        graph_all = False
        includedTicker = ""
        i = 0
        graph_tickers = []
        while((includedTicker!="" and includedTicker!="f") or i == 0):
                includedTicker = input("Add ETF to graph (enter ticker, or f to finish): ")
                graph_tickers.append(includedTicker)
                i = i+1
else:
        graph_all = True
#Choose which individual ETFs to include on the graph. If graph_all is "yes", all ETFs will be graphed without labels

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


def solve_weights(R, C, B, rf):
        def fitness(W, R, C, B, rf):
                mean, var = port_mean_var(W, R, C)
                beta, var = port_mean_var(W, B, C)      # calculate mean/variance of the portfolio
                util = (mean - rf) / sqrt(var)          # utility = Sharpe ratio
                return 1/util                                           # maximize the utility, minimize its inverse value
        n = len(R)
        W = ones([n])/n                                         # start optimization with equal weights
        b_ = [(0.,1.) for i in range(n)]        # weights for boundaries between 0%..100%. No leverage, no shorting
        c_ = ({'type':'eq', 'fun': lambda W: sum(W)-1. })       # Sum of weights must be 100%
        optimized = scipy.optimize.minimize(fitness, W, (R, C, B, rf), method='SLSQP', constraints=c_, bounds=b_, options={'maxiter':100000, 'ftol': 1e-6} )
        if not optimized.success:
                raise BaseException(optimized.message)
        return optimized.x

def solve_frontier(R, C, B, rf):
    def fitness(W, R, C, B, r):
            # For given level of return r, find weights which minimizes
            # portfolio variance.
            mean, var = port_mean_var(W, R, C)
            # Big penalty for not meeting stated portfolio return effectively serves as optimization constraint
            penalty = 50*abs(mean-r)

            return var + penalty
    frontier_mean, frontier_var, frontier_weights, frontier_beta, frontier_sharpe, frontier_treynor = [], [], [], [], [], []
    n = len(R)      # Number of assets in the portfolio
    for r in linspace(min(R), max(R), num=400): # Iterate through the range of returns on Y axis
            W = ones([n])/n         # start optimization with equal weights
            b_ = [(0,1) for i in range(n)]
            c_ = ({'type':'eq', 'fun': lambda W: sum(W)-1. })
            optimized = scipy.optimize.minimize(fitness, W, (R, C, B, r), method='SLSQP', constraints=c_, bounds=b_,options={'maxiter':100000})
            if not optimized.success:
                    raise BaseException(optimized.message)
            # add point to the min-var frontier [x,y] = [optimized.x, r]
            frontier_mean.append(r)                                                 # return
            frontier_var.append(port_var(optimized.x, C))   # min-variance based on optimized weights
            frontier_beta.append(port_mean(optimized.x,B))
            frontier_weights.append(optimized.x)
            frontier_sharpe.append((r-rf)/(port_var(optimized.x,C)**.5))
            frontier_treynor.append((r-rf)/port_mean(optimized.x,B))
    return array(frontier_mean), array(frontier_var), array(frontier_beta), array(frontier_sharpe), array(frontier_treynor),frontier_weights






now = datetime.date.today()
date = str(now)


outputFile = []

print("Pulling Market Data...")
spy = yf.Ticker("SPY")
if(set_period_manually != True):
        marketHist = spy.history(period=time_period,interval="1d")
else:
        marketHist = spy.history(start=start_date,end=end_date,interval="1d")
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
        if(set_period_manually != True):
                fiHist = agg.history(period=time_period,interval="1d")
        else:
                fiHist = agg.history(start=start_date,end=end_date,interval="1d")
        fiCloses = fiHist['Close'].tolist()
        fullLength = len(fiCloses)
        temp = []
        for j in range(0, len(fiCloses)):
                if(j%21 == 0):
                        temp.append(fiCloses[j])
        fiCloses=temp
        fiReturns = pd.Series(fiCloses).pct_change().tolist()[1:]

        gld = yf.Ticker("GLD")
        if(set_period_manually != True):
                commHist = gld.history(period=time_period,interval="1d")
        else:
                commHist = gld.history(start=start_date,end=end_date,interval="1d")
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



print("Pulling ETF Data...")
reader = csv.reader(open("etfAUM.csv", "r",encoding="utf8", errors='ignore'), delimiter=",")
etfs = list(reader)
assetClass = list(np.transpose(etfs)[3])
aum = list(np.transpose(etfs)[2])
etfs = list(np.transpose(etfs)[0])
etfNames = list(np.transpose(etfs)[1])
temp = []

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

    if(set_period_manually != True):
        hist = company.history(period=time_period, interval="1d")
    else:
        hist = company.history(start=start_date, end=end_date, interval="1d")
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

print("Computing returns and covariances...")
temp = []
marketCapSum = sum(marketCaps)
for marketCap in marketCaps:
        temp.append(marketCap/marketCapSum)
marketCapWeights = temp




if(graph_all == True):
    graph_tickers = tickers


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
if(y_variable == "Implied Excess Return"):
        returns_annual = excess

variances = []
for j in range(0, len(cov_monthly)):
    variances.append(cov_annual[j][j])
prices=np.transpose([np.append("Ticker", tickers), np.append("2-Year Average Annual Returns", returns_annual), np.append("Variance of Daily Returns" ,variances)])

print("Finding Efficient Frontier...")
port_returns = []
port_volatility = []
port_beta = []
stock_weights = []
sharpe_ratio = []
treynor_ratio = []
# set the number of combinations for imaginary portfolios
num_assets = len(tickers)
num_portfolios = 50000

W = solve_weights(returns_annual, cov_annual, beta_annual, risk_free_rate)
mean, var = port_mean_var(W, returns_annual, cov_annual)  
f_mean, f_var, f_beta, f_sharpe, f_treynor, f_weights = solve_frontier(returns_annual, cov_annual, beta_annual, risk_free_rate)

color = 'black'
n = len(tickers)

std_filt = []
ret_filt = []
beta_filt = []
tickers_filt = []
sharpe_filt = []
treynor_filt = []
for i in range(n):
    if(tickers[i] in graph_tickers):
        std_filt.append(cov_annual[i,i]**.5)
        ret_filt.append(returns_annual[i])
        beta_filt.append(beta_annual[i])
        tickers_filt.append(tickers[i])
        sharpe_filt.append((returns_annual[i]-risk_free_rate)/(cov_annual[i,i]**.5))
        treynor_filt.append((returns_annual[i]-risk_free_rate)/beta_annual[i])


f_std = f_var**.5

# temp = [f_std, f_mean]
# temp = np.sort(temp, axis=0)
# f_std = []
# f_mean = []
# for i in range(0, len(temp[0])):
#     if(i == 0):
#         current = temp[0][i]
#         f_std.append(temp[0][0])
#         f_mean.append(temp[0][1])
#     else:
#         print(temp[0][i])
#         print(current)
#         if(temp[0][i]-.01 > current):
#             current = temp[0][i]
#             f_std.append(temp[0][i])
#             f_mean.append(temp[1][i])

# print(f_std)
# print(f_mean)

if(y_variable == "Sharpe Ratio"):
    f_y = f_sharpe
    y_filt = sharpe_filt
elif(y_variable == "Nominal Return"):
    f_y = f_mean
    y_filt = ret_filt
elif(y_variable == "Implied Excess Return"):
    f_y = f_mean
    y_filt = ret_filt
elif(y_variable == "Real Return"):
    f_y = np.asarray(f_mean)*inflation_factor
    y_filt = np.asarray(ret_filt)*inflation_factor
elif(y_variable == "Treynor Ratio"):
    f_y = f_treynor
    y_filt = treynor_filt
if(x_variable == "Volatility"):
    f_x = f_std
    x_filt = std_filt
elif(x_variable == "Beta"):
    f_x = f_beta
    x_filt = beta_filt



portfolio = {'Nominal Return': ret_filt,
                'Real Return': np.asarray(ret_filt)*inflation_factor,
                'Implied Excess Return': ret_filt,
             'Volatility': std_filt,
             'Sharpe Ratio': sharpe_filt,
             'Treynor Ratio': treynor_filt,
             'Beta': beta_filt}

df = pd.DataFrame(portfolio)


column_order = ['Nominal Return', 'Real Return', 'Implied Excess Return', 'Volatility', 'Sharpe Ratio', 'Treynor Ratio', 'Beta']
df = df[column_order]
print("Graphing Efficient Frontier")
plt.style.use('seaborn-dark')
if(gradient_variable != "None"):
        df.plot.scatter(x=x_variable, y=y_variable, c=gradient_variable,
                cmap='RdYlGn',figsize=(10,8), grid=True,edgecolors='black')
else:
        df.plot.scatter(x=x_variable, y=y_variable, figsize=(10,8), grid=True,edgecolors='black')
if(graph_all != True):
        for i in range(len(x_filt)):                                                                              # draw labels
                plt.text(x_filt[i], y_filt[i], '  %s'%tickers_filt[i], verticalalignment='center', color=color)
plt.plot(f_x, f_y, color='blue')
# plt.plot(f_x, f_y+f_x, color='green')  
# plt.plot(f_x, f_y-f_x, color='red')
if(historical == True):
        plt.title('Historical Efficient Frontier: ' + date) 
else:
        plt.title('Black-Litterman Efficient Frontier: ' + date)                                 
plt.xlabel(x_variable)
plt.ylabel(y_variable)
plt.grid(True)
plt.savefig("efficient-frontier.png")
plt.show()







if(historical == True):
        for single_portfolio in range(num_portfolios):
                weights = np.random.random(num_assets)
                weights /= np.sum(weights)
                returns = np.dot(weights, returns_annual)
                beta = np.dot(weights, beta_annual)
                port_beta.append(beta)
                volatility = np.sqrt(np.dot(weights.T, np.dot(cov_annual, weights)))
                port_returns.append(returns)
                sharpe = (returns-risk_free_rate) / volatility
                sharpe_ratio.append(sharpe)
                treynor = (returns-risk_free_rate) / beta
                treynor_ratio.append(treynor)
                port_volatility.append(volatility)
                stock_weights.append(weights)

        # a dictionary for Returns and Risk values of each portfolio
        portfolio = {'Nominal Return': port_returns,
                'Real Return': port_returns*inflation_factor,
                'Volatility': port_volatility,
                'Sharpe Ratio': sharpe_ratio,
                'Treynor Ratio': treynor_ratio,
                'Beta': port_beta}

        # extend original dictionary to accomodate each ticker and weight in the portfolio
        for counter,symbol in enumerate(tickers):
                portfolio[symbol+' Weight'] = [Weight[counter] for Weight in stock_weights]


        df = pd.DataFrame(portfolio)


        column_order = ['Nominal Return', 'Volatility', 'Sharpe Ratio', 'Treynor Ratio', 'Beta'] + [stock+' Weight' for stock in tickers]
        df = df[column_order]

        min_volatility = df['Volatility'].min()
        max_sharpe = df['Sharpe Ratio'].max()

        # use the min, max values to locate and create the two special portfolios
        sharpe_portfolio = df.loc[df['Sharpe Ratio'] == max_sharpe]
        min_variance_port = df.loc[df['Volatility'] == min_volatility]

        # plot frontier, max sharpe & min Volatility values with a scatterplot
        plt.style.use('seaborn-dark')
        if(gradient_variable == "None"):
                df.plot.scatter(x=x_variable, y=y_variable,edgecolors='black', figsize=(10, 8), grid=True)
        else:
                df.plot.scatter(x=x_variable, y=y_variable, c=gradient_variable,
                        cmap='RdYlGn', edgecolors='black', figsize=(10, 8), grid=True)
        plt.xlabel(x_variable)
        plt.ylabel(y_variable)
        plt.title('Simulated Random-Weighted Portfolios: ' + date)
        plt.savefig('simulated-portfolios.png')
        plt.show()



# with open('etf-data.csv', "w", newline='') as f:
#     writer = csv.writer(f)
#     writer.writerows(prices)   

# covariance = [[" "]]
# for i in range(0, len(cov_monthly)):
#     covariance[0].append(tickers[i])
# for j in range(0, len(cov_monthly)):
#     temp = []
#     temp.append(tickers[j])
#     for k in range(0, len(cov_monthly)):
#         temp.append(cov_monthly[j][k])
#     covariance.append(temp)

# with open('covariance-matrix.csv', "w", newline='') as f:
#     writer = csv.writer(f)
#     writer.writerows(covariance)

# file_list = [
#     'efficient-frontier.png',
#     'etf-data.csv',
#     'covariance-matrix.csv'
# ]
# commit_message = 'Add graph'
# repo.index.add(file_list)
# repo.index.commit(commit_message)
# origin = repo.remote('origin')
# origin.push(force=True)

# print(min_variance_port.T)
# print(sharpe_portfolio.T)
 



