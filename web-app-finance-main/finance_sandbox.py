import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from flask import Flask, render_template, request
from prophet import Prophet
import base64
import io
import random 
import yfinance as yf
from fredapi import Fred

ticker_names = pd.read_csv('https://www.alphavantage.co/query?function=LISTING_STATUS&apikey=demo')['symbol']
ticker_names = np.append(ticker_names, ['ETH-USD', 'BTC-USD'])
fred = Fred(api_key = '09ae017da9a8f43be50d4c6e6914c33d')
risk_free_rate = fred.get_series_latest_release('GS10')[-1] / 100


tickers = ['AAPL', 'TSLA', 'AMZN']



df = yf.download(tickers = tickers,
                start=datetime.today()-timedelta(days=365*12),
                end=datetime.today())

market_return = yf.download(tickers = '^GSPC',
                start=datetime.today()-timedelta(days=365*12),
                end=datetime.today())

market_return_daily = market_return['Adj Close'].pct_change().dropna()
market_return = market_return['Adj Close'].resample('YE').ffill().dropna().pct_change()
market_return_avg = market_return.mean()
daily_return = df['Adj Close'].pct_change().dropna()
yearly_data = df['Adj Close'].resample('YE').ffill().dropna()
annual_return = yearly_data.pct_change()
tickers = annual_return.columns.tolist()

n_assets = len(tickers)
n_sims = 10000
weights = np.zeros((n_sims, n_assets))
p_expReturn = np.zeros(n_sims)
p_std = np.zeros(n_sims)

for i in range(n_sims):
    random_weights = np.random.rand(n_assets)
    random_weights /= random_weights.sum()
    weights[i, ] = random_weights
    p_expReturn[i] = random_weights @ annual_return.mean()
    p_std[i] = np.sqrt(random_weights.T @ annual_return.cov() @ random_weights)

sharpe_ratios = (p_expReturn - risk_free_rate) / p_std

max_sharpe_ratio = max(sharpe_ratios)
optimal_return = p_expReturn[sharpe_ratios.argmax()]
optimal_std = p_std[sharpe_ratios.argmax()]
optimal_weights = np.round(weights[sharpe_ratios.argmax(), ], 3)

optimal_portfolio = pd.DataFrame({'Tickers': tickers, 'Weights': optimal_weights})

portfolio_data = pd.DataFrame({
    'p_std': p_std,
    'p_expReturn': p_expReturn,
    'sharpe_ratios': sharpe_ratios
})

daily_return
market_return_daily

############## SML #####################
min_length = min(len(market_return_daily), len(daily_return))
market_return_daily = market_return_daily[-min_length:]

# Calculate betas for each stock dynamically based on the columns in annual_return
beta_i = np.array([
    np.cov(market_return_daily[1:], daily_return[ticker].values[-min_length+1:])[0, 1]
    for ticker in tickers
]) / np.var(market_return_daily)

returns_i = annual_return.mean()

portfolio_beta = np.sum(beta_i * optimal_weights)

beta_x = np.arange(beta_i.min() - 1, beta_i.max() + 1, 0.01)
market_return_avg = market_return.mean()
SML = risk_free_rate + beta_x * (market_return_avg - risk_free_rate)

fig, ax4 = plt.subplots(figsize=(10, 5))
ax4.plot(beta_x, SML, label="Security Market Line", color='steelblue', linewidth=2)
ax4.scatter(portfolio_beta, optimal_return, color='red', s=50, zorder=5)
ax4.scatter(beta_i, returns_i, color='black')

for i, label in enumerate(tickers):
    ax4.text(beta_i[i], returns_i[label]+optimal_return*.1, label, fontsize=9, color='blue')

ax4.set_title("Security Market Line")
ax4.set_xlabel("Beta")
ax4.set_ylabel("E[R]")

beta_i


































































def finance(tickers):
    df = yf.download(tickers = tickers,
                    start=datetime.today()-timedelta(days=365*12),
                    end=datetime.today())

    market_return = yf.download(tickers = '^GSPC',
                    start=datetime.today()-timedelta(days=365*12),
                    end=datetime.today())

    market_return = market_return['Adj Close'].resample('YE').ffill().dropna().pct_change()
    market_return_avg = market_return.mean()
    #daily_return = df['Adj Close'].pct_change().dropna()
    yearly_data = df['Adj Close'].resample('YE').ffill().dropna()
    annual_return = yearly_data.pct_change()
    tickers = annual_return.columns.tolist()

    n_assets = len(tickers)
    n_sims = 10000
    weights = np.zeros((n_sims, n_assets))
    p_expReturn = np.zeros(n_sims)
    p_std = np.zeros(n_sims)

    for i in range(n_sims):
        random_weights = np.random.rand(n_assets)
        random_weights /= random_weights.sum()
        weights[i, ] = random_weights
        p_expReturn[i] = random_weights @ annual_return.mean()
        p_std[i] = np.sqrt(random_weights.T @ annual_return.cov() @ random_weights)

    sharpe_ratios = (p_expReturn - risk_free_rate) / p_std

    max_sharpe_ratio = max(sharpe_ratios)
    optimal_return = p_expReturn[sharpe_ratios.argmax()]
    optimal_std = p_std[sharpe_ratios.argmax()]
    optimal_weights = np.round(weights[sharpe_ratios.argmax(), ], 3)

    optimal_portfolio = pd.DataFrame({'Tickers': tickers, 'Weights': optimal_weights})

    portfolio_data = pd.DataFrame({
        'p_std': p_std,
        'p_expReturn': p_expReturn,
        'sharpe_ratios': sharpe_ratios
    })

    # cml_x = np.arange(0, max(p_std)/2, 0.01)
    # cml_y = risk_free_rate + cml_x * (optimal_return - risk_free_rate) / optimal_std
    # cml = pd.DataFrame({
    #     'x': cml_x,
    #     'y': cml_y
    # })

    ############## Efficiency Frontier #####################
    fig, ax3 = plt.subplots(figsize=(10, 5))
    scatter = ax3.scatter(portfolio_data['p_std'], portfolio_data['p_expReturn'], c=portfolio_data['sharpe_ratios'], cmap='Blues', s=2)
    #ax3.plot(cml['x'], cml['y'], 'r--', linewidth=1)  # CML line
    ax3.plot([0,optimal_std] , [optimal_return, optimal_return] , color='grey', linestyle='--', lw=0.7)
    ax3.plot([optimal_std,optimal_std] , [0, optimal_return] , color='grey', linestyle='--', lw=0.7)
    ax3.scatter(optimal_std, optimal_return, color='red', s=50)

    ax3.set_title("Portfolio Efficiency Frontier", fontsize=15)
    ax3.set_xlabel("Risk", fontsize=10)
    ax3.set_ylabel("Return", fontsize=10)
    fig.colorbar(scatter, label='Sharpe Ratio')
    ax3.set_xlim(0.6*min(p_std), max(p_std))
    ax3.set_ylim(0.6*min(p_expReturn), max(p_expReturn))


    ############## SML #####################
    min_length = min(len(market_return), len(annual_return))
    market_return = market_return[-min_length:]

    # Calculate betas for each stock dynamically based on the columns in annual_return
    beta_i = np.array([
        np.cov(market_return[1:], annual_return[ticker].values[-min_length+1:])[0, 1]
        for ticker in tickers
    ]) / np.var(market_return)

    returns_i = annual_return.mean()

    portfolio_beta = np.sum(beta_i * optimal_weights)

    beta_x = np.arange(beta_i.min() - 1, beta_i.max() + 1, 0.01)
    market_return_avg = market_return.mean()
    SML = risk_free_rate + beta_x * (market_return_avg - risk_free_rate)

    fig, ax4 = plt.subplots(figsize=(10, 5))
    ax4.plot(beta_x, SML, label="Security Market Line", color='steelblue', linewidth=2)
    ax4.scatter(portfolio_beta, optimal_return, color='red', s=50, zorder=5)
    ax4.scatter(beta_i, returns_i, color='black')

    for i, label in enumerate(tickers):
        ax4.text(beta_i[i], returns_i[label]+optimal_return*.1, label, fontsize=9, color='blue')

    ax4.set_title("Security Market Line")
    ax4.set_xlabel("Beta")
    ax4.set_ylabel("E[R]")

    print(f"Optimal Portfolio Sharpe Ratio: {max_sharpe_ratio:.2f}")
    print(f"Optimal Portfolio Return: {optimal_return:.2f}")
    print(f"Optimal Portfolio STD: {optimal_std:.2f}")
    print(f"Optimal Portfolio Weights: ")
    print(optimal_portfolio)


finance(['BTC-USD', 'ETH-USD', 'AAPL'])


















def price_fcst(name, period, period_unit):

    if period_unit == "year":
        period = period*365
    elif period_unit == "month":
        period = period*30

    df = yf.download(tickers = {name},
                 start=datetime.today()-timedelta(days=365*5),
                 end=datetime.today())
    df = df.loc[:, ['Adj Close']]
    df.columns = ['y']
    df.index.rename('ds', inplace=True)

    fig, ax1 = plt.subplots(figsize=(10, 5))
    ax1.plot(df.index, df.y, color='firebrick')
    ax1.set_title(f"The Stock Price of {name}", size='x-large', color='blue')
    ax1.set_xlabel('Date')
    ax1.set_ylabel('Price (USD)')
    ax1.plot(df.index[-1], df.y[-1], 'bo')
    ax1.text(df.index[-1] + timedelta(100), df.y[-1], f'${df.y[-1]:.2f}')

    df = df.reset_index()
    model = Prophet()
    model_fit = model.fit(df)
    future = model_fit.make_future_dataframe(periods=int(period), freq='D')

    pred = model_fit.predict(future)

    final_date = pred.ds.iloc[-1].strftime("%m-%d-%Y")
    future_final_price = pred.yhat.iloc[-1]
    present_price = df.y.iloc[-1]
    slope = (future_final_price - present_price) / (int(period))
    future_pctch = 100 * (future_final_price - present_price) / present_price
    past_five_pctch = 100 * (df.y.iloc[-1] - df.y.iloc[0]) / df.y.iloc[0]

    future_pctch_bool = "decrease"
    past_five_pctch_bool = "decreased"
    if future_pctch > 0:
        future_pctch_bool = 'increase'
    if past_five_pctch > 0:
        past_five_pctch_bool = 'increased'

    fig, ax = plt.subplots(figsize=(10, 5))
    model.plot(pred, ax=ax)
    ax.set_title(f'Forecast Model of {name} Closing Prices')
    ax.plot(pred.ds.iloc[-1], future_final_price, 'bo')
    ax.text(pred.ds.iloc[-1] + timedelta(days=120), future_final_price, f"${future_final_price:.2f}")
    ax.set_xlabel('Date')
    ax.set_ylabel('Closing Price (USD)')

    if period_unit == "year":
        period = period/365
    elif period_unit == "month":
        period = period/30

    return (f"{name} current price is ${df.y.iloc[-1]:.2f}."
            f"It's closing price has {past_five_pctch_bool} by {past_five_pctch:.2f}% in the past five years "
            f"and is predicted to {future_pctch_bool} by {future_pctch:.2f}% in the next {int(period)} {period_unit}(s).\n\n"
            f" The trend has an average rate of {slope:.2f}. Meaning, on average, for every calendar day that passes, the price of {name} is predicted to {future_pctch_bool} by {slope:.2f} USD. \n\nThe predicted price for {name} on {final_date} is ${pred.yhat.iloc[-1]:.2f}.")


price_fcst('BTC-USD', 1, 'month')















