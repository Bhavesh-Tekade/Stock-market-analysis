# Stock Market Data Analysis Project

# 1. Import Libraries
import pandas as pd
from pandas import Series, DataFrame
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style('whitegrid')
from datetime import datetime
import pandas_datareader as web
from __future__ import division

# 2. Load Stock Data
tech_list = ['AAPL','GOOG','MSFT','AMZN']
end = datetime.now()
start = datetime(end.year - 1, end.month, end.day)
for stock in tech_list:
    globals()[stock] = web.DataReader(stock, 'iex', start, end)

# 3. Basic Info and Summary
print(AAPL.head())
print(AAPL.describe())
print(AAPL.info())

# 4. Visualization of Close and Volume
AAPL['close'].plot(legend=True, figsize=(12,6))
plt.show()
AAPL['volume'].plot(legend=True, figsize=(10,4))
plt.show()

# 5. Moving Averages
ma_day = [10,20,50]
for ma in ma_day:
    column_name = f"MA for {ma} days"
    AAPL[column_name] = AAPL['close'].rolling(ma).mean()
AAPL[['close','MA for 10 days','MA for 20 days','MA for 50 days']].plot(figsize=(12,6))
plt.show()

# 6. Daily Return Analysis
AAPL['Daily Return'] = AAPL['close'].pct_change()
AAPL['Daily Return'].plot(figsize=(12,5), legend=True, linestyle='--', marker='o')
plt.show()
sns.histplot(AAPL['Daily Return'].dropna(), bins=100, color='blue', kde=True)
plt.show()

# 7. Comparison of Multiple Stocks
closing_df = pd.concat([
    AAPL['close'].rename("AAPL_close"),
    GOOG['close'].rename("GOOG_close"),
    MSFT['close'].rename("MSFT_close"),
    AMZN['close'].rename("AMZN_close")
], axis=1)
tech_returns = closing_df.pct_change()

# 8. Pairwise Comparison
sns.jointplot('GOOG_close','GOOG_close',tech_returns, kind='scatter', color='seagreen')
plt.show()
sns.jointplot('GOOG_close','MSFT_close',tech_returns, kind='scatter')
plt.show()
sns.pairplot(tech_returns.dropna())
plt.show()

# 9. PairGrid for Daily Return
returns_fig = sns.PairGrid(tech_returns.dropna())
returns_fig.map_upper(plt.scatter, color='purple')
returns_fig.map_lower(sns.kdeplot, cmap='cool_d')
returns_fig.map_diag(plt.hist, bins=30)
plt.show()

# 10. PairGrid for Closing Prices
returns_fig = sns.PairGrid(closing_df)
returns_fig.map_upper(plt.scatter, color='purple')
returns_fig.map_lower(sns.kdeplot, cmap='cool_d')
returns_fig.map_diag(plt.hist, bins=30)
plt.show()

# 11. Heatmap of Correlation
corr = tech_returns.dropna().corr()
mask = np.triu(np.ones_like(corr, dtype=bool))
cmap = sns.diverging_palette(220, 10, as_cmap=True)
plt.figure(figsize=(11,9))
sns.heatmap(corr, mask=mask, cmap=cmap, annot=True)
plt.show()

# 12. Risk Analysis
rets = tech_returns.dropna()
area = np.pi*20
plt.scatter(rets.mean(), rets.std(), alpha=0.5, s=area)
plt.xlabel('Expected returns')
plt.ylabel('Risk')
plt.ylim([0.01,0.025])
plt.xlim([-0.003,0.005])
for label, x, y in zip(rets.columns, rets.mean(), rets.std()):
    plt.annotate(label, xy=(x,y), xytext=(50,50),
                 textcoords='offset points', ha='right', va='bottom',
                 arrowprops=dict(arrowstyle='-', connectionstyle='arc3,rad=-0.3'))
plt.show()

# 13. Value at Risk (Bootstrap Method)
sns.histplot(AAPL['Daily Return'].dropna(), bins=100)
plt.show()
emp = rets['AAPL_close'].quantile(0.05)
print("The 0.05 empirical quantile of daily returns is at", emp)

# 14. Monte Carlo Simulation
# Parameters
days = 365
deltaT = 1 / days
mu = rets.mean()['GOOG_close']
sigma = rets.std()['GOOG_close']

# Simulation Function
def monte_carlo_simulation(start_price, days, mu, sigma):
    price = np.zeros(days)
    price[0] = start_price
    for x in range(1, days):
        shock = np.random.normal(loc=mu * deltaT, scale=sigma * np.sqrt(deltaT))
        drift = mu * deltaT
        price[x] = price[x-1] + (price[x-1] * (drift + shock))
    return price

# Plot Simulations
start_price = 1027.27
for run in range(100):
    plt.plot(monte_carlo_simulation(start_price, days, mu, sigma))
plt.xlabel("Days")
plt.ylabel("Price")
plt.title('Monte Carlo Analysis for Google Stock\n')
plt.show()

# Distribution of Final Prices
runs = 10000
simulations = np.zeros(runs)
for run in range(runs):
    simulations[run] = monte_carlo_simulation(start_price, days, mu, sigma)[-1]
q = np.percentile(simulations, 1)
plt.hist(simulations, bins=200)
plt.figtext(0.6, 0.8, f"Start price: ${start_price:.2f}")
plt.figtext(0.6, 0.7, f"Mean final price: ${simulations.mean():.2f}")
plt.figtext(0.6, 0.6, f"VaR(0.99): ${start_price - q:.2f}")
plt.figtext(0.15, 0.6, f"q(0.99): ${q:.2f}")
plt.axvline(x=q, linewidth=4, color='r')
plt.title(f"Final price distribution for Google Stock after {days} days\n", weight='bold')
plt.show()
