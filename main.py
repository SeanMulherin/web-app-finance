import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend

from flask import Flask, render_template, request
import matplotlib.pyplot as plt
import base64
import io
import numpy as np
import pandas as pd
import seaborn as sns
from datetime import datetime, timedelta
import datetime as dt
import yfinance as yf

from prophet import Prophet
from prophet.diagnostics import performance_metrics

app = Flask(__name__, static_url_path='/static')


def price_fcst(name, period):
    df = yf.download(tickers = {name},
                 start=datetime.today()-timedelta(days=365*5),
                 end=datetime.today())
    df = df.loc[:, ['Close']]
    df.columns = ['y']
    df.index.rename('ds', inplace=True)

    fig, ax1 = plt.subplots(figsize=(10, 5))
    ax1.plot(df.index, df.y, color='firebrick')
    ax1.set_title(f"The Stock Price of {name}", size='x-large', color='blue') # add name var
    ax1.set_xlabel('Date')
    ax1.set_ylabel('Price (USD)')
    ax1.plot(df.index[-1], df.y[-1], 'bo')
    ax1.text(df.index[-1] + timedelta(100), df.y[-1], f'${df.y[-1]:.2f}')

    plot_bytes_io1 = io.BytesIO()
    plt.savefig(plot_bytes_io1, format='png')
    plot_bytes_io1.seek(0)
    plot_base641 = base64.b64encode(plot_bytes_io1.getvalue()).decode()
    plot_html1 = f'<img src="data:image/png;base64,{plot_base641}" alt="plot" />'

    plt.close()

    df = df.reset_index()
    model = Prophet()
    model_fit = model.fit(df)
    future = model_fit.make_future_dataframe(periods=365*period, freq='D')

    pred = model_fit.predict(future)

    future_final_price = pred.yhat.iloc[-1]
    present_price = pred.yhat.iloc[-365*period]
    slope = (future_final_price - pred.yhat.iloc[-365*period]) / (365*period)
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
    ax.text(pred.ds.iloc[-1] + timedelta(days=50), future_final_price, f"${future_final_price:.2f}")
    ax.set_xlabel('Date')
    ax.set_ylabel('Closing Price (USD)')

    plot_bytes_io = io.BytesIO()
    plt.savefig(plot_bytes_io, format='png')
    plot_bytes_io.seek(0)
    plot_base64 = base64.b64encode(plot_bytes_io.getvalue()).decode()
    plot_html = f'<img src="data:image/png;base64,{plot_base64}" alt="plot" />'

    plt.close()

    components_fig = model.plot_components(pred, figsize=(10, 7))
    components_bytes_io = io.BytesIO()
    components_fig.savefig(components_bytes_io, format='png')
    components_bytes_io.seek(0)
    components_base64 = base64.b64encode(components_bytes_io.getvalue()).decode()
    components_html = f'<img src="data:image/png;base64,{components_base64}" alt="components plot" />'

    plt.close(components_fig)

    final_date = pred.ds.iloc[-1].strftime("%m-%d-%Y")
    print(f"The trend has an average rate of {slope:.4f}")
    print(f"The predicted price for {name} on {final_date} is ${pred.yhat.iloc[-1]:.2f}")

    return (f"{name} closing prices have {past_five_pctch_bool} by {past_five_pctch:.2f}% in the past five years "
            f"and are predicted to {future_pctch_bool} by {future_pctch:.2f}% in the next {period} year(s).", 
            plot_html1, plot_html, components_html)


@app.route('/')
def index():
    return render_template('finance.html')

@app.route('/forecast', methods=['POST'])
def forecast():
    ticker = request.form['ticker']
    ticker = ticker.strip()
    ticker = ticker.upper()
    period = int(request.form['period'])

    if period < 0:
        result = f"ERROR: FORECAST LENGTH MUST BE POSITIVE. YOU INPUT {period} AS YOUR FORECAST LENGTH. PLEASE GO BACK A PAGE AND TRY AGAIN."
        return render_template('result.html', result=result)
    else:
        result,plot_html1, plot_html, components_html = house_fcst(ticker, period)
        return render_template('result.html', result=result, plot_html1=plot_html1, plot_html=plot_html, components_html=components_html)


if __name__ == "__main__":
    app.run(debug=True)
