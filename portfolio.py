import numpy as np
import riskfolio as rp
import yfinance as yf 
import matplotlib.pyplot as plt 
import pandas as pd 
import streamlit as st
import plotly.graph_objs as go
import quantstats as qs
import warnings
import datetime 
warnings.filterwarnings("ignore")

from neuralprophet import NeuralProphet
from ydata_profiling import ProfileReport
from streamlit_pandas_profiling import st_profile_report


with st.sidebar:
    choice = st.radio("Select One:", [
        "Welcome!",
        "ARIMA Price Prediction",
        "Portfolio EDA & Optimization",  
        "Next Steps",
    ])

if choice == "Welcome!":
    with st.expander("Introduction to stock market quantitative analaysis in Python", expanded=False):
        st.write("Steps:")
        st.write("ARIMA Model Price Prediction")
        st.write("- Utilzie ARIMA & [Neural Prophet](https://neuralprophet.com/) to explore trend analysis of your security of choice")
        st.write("Portfolio EDA & Optimization:")
        st.write("- Explore portfolio optimizing strategies such as Sharpe ratio's and backtesting your optimal portfolio against $SPY")
        st.write("Next Steps:")
        st.write("- Navigate to this [jupyter notebook](https://colab.research.google.com/drive/11NqRw2AIDS8mZ-kjCbjTc36-T_7tnR34#scrollTo=zLIBdzWjdqq3) to expand your knowledge of market algorithmic analaysis")

    st.subheader("Created by Jonathan Hofmann ---> [github](https://github.com/hofmannj0n) | [portfolio](https://www.jhofmann.me/)")
    st.write("")
    st.write("Tutorial:")
  
if choice == "ARIMA Price Prediction":
    st.title("Stock Market Prediction with ARIMA")

    image_url = "https://slideplayer.com/slide/4283195/14/images/69/ARIMA+models+for+time+series+data.jpg"

    link_url = "https://www.investopedia.com/terms/a/autoregressive-integrated-moving-average-arima.asp"

    # HTML code with image sizing
    html_str = f"""
    <a href="{link_url}" target="_blank">
        <img src="{image_url}" alt="Alt text" width="700" height="500"/>
    </a>
    """

    st.markdown(html_str, unsafe_allow_html=True)
    st.divider()

    st.info("""
The acronym ARIMA stands for AutoRegressive Integrated Moving Average, and it is a widely used forecasting method in financial analysis because it captures essential aspects of financial / time series data such as trends, seasonality, and random fluctuations. 

AutoRegressive (AR): This component leverages the dependencies between an observation and a certain number of lagged observations. It's like recognizing that weather conditions from previous days can influence the forecast for today.

Integrated (I): This component involves differencing the actual observations to make the time series stationary. This means subtracting the previous observation from the current one to deal with trends in the data.  Going off the weather example, it could involve estimating how much the weather changes day by day, rather than the total amount of rainfall.

Moving Average (MA): This component models the relationship between an observation and a residual error from a moving average model applied to lagged observations. Basically smooths out noise in the data to focus on more substantial trends and cycles.

The parameters of an ARIMA model are usually denoted as 
p,d,q:

p (Lags of the auto-regressive model): This parameter deals with the lag variables.  Continuing with the weather forecasting analogy, you might look at how the past few days weather can give you insights into tomorrow's weather.

d (Degree of differencing): This involves the number of times the data needs to be differenced to become stationary. It reflects how comparing changes from day to day can help model the data more effectively, especially if the weather has been following a consistent pattern.

q (Order of the moving average model): This is about the size of the moving average window and is used to smooth out short term fluctuations and highlight longer term trends or cycles in the dataset.
            """)

    st.divider()
    st.subheader("Select a Stock and Utilize ARIMA Forecasting:")

    # stock input variables 
    st.write("[List of all Stock Symbols](https://stockanalysis.com/stocks/)")
    stock = st.text_input("Enter Stock Symbol", "AAPL")
    start_date = st.date_input("Start Date", datetime.datetime(2015, 1, 1))
    end_date = st.date_input("End Date", datetime.datetime.now())

    if st.button("Predict Stock Close Price"):

        stocks = yf.download(stock, start=start_date, end=end_date)

        # manipulating df to neuralprophet specifications
        stocks.reset_index(inplace=True)
        stocks = stocks[['Date', 'Close']]
        stocks.columns = ['ds', 'y']

        # initializing the model
        m = NeuralProphet()
        m.fit(stocks, freq='B', epochs=100)

        future = m.make_future_dataframe(stocks, periods=365)
        forecast = m.predict(future)
        actual_prediction = m.predict(stocks)


        fig = go.Figure()

        # actual past data
        fig.add_trace(go.Scatter(x=stocks['ds'], y=stocks['y'], mode='lines', name='Actual Close Price', line=dict(color='green')))
        
        # predictions on past data
        fig.add_trace(go.Scatter(x=actual_prediction['ds'], y=actual_prediction['yhat1'], mode='lines', name='Predicted Close Price', line=dict(color='red')))
        
        # future predictions
        fig.add_trace(go.Scatter(x=forecast['ds'], y=forecast['yhat1'], mode='lines', name='Future Predictions', line=dict(color='blue')))
        
        # Update plot 
        fig.update_layout(title='Stock Market Predictions for ' + stock,
                        xaxis_title='Date',
                        yaxis_title='Stock Price',
                        legend_title='Legend',
                        xaxis_rangeslider_visible=False)

        st.plotly_chart(fig, use_container_width=True)
        st.divider()
        st.write("Observed, Trend, Seasonality:")
        st.plotly_chart(m.plot(forecast), use_container_width=True)
        st.plotly_chart(m.plot_components(forecast), use_container_width=True)
        st.divider()
        st.write("Profile Report:")
        profile = ProfileReport(stocks, title="Profiling Report")
        st_profile_report(profile)

if choice == "Portfolio EDA & Optimization":

    # common stocks 
    stocks = ["AAPL", "GOOGL", "IBM", "MSFT", "VIX", "VOO", "QQQ", "TSLA", "JPM", "AMZN", "VZ", "NVDA", "BAC", "SBUX", "NKE", "MA", "PLTR"]    
    
    st.subheader("Understanding The Sharpe Ratio:")

    image_url = "https://cdn.corporatefinanceinstitute.com/assets/sharpe-ratio.png"

    link_url = "https://corporatefinanceinstitute.com/resources/career-map/sell-side/risk-management/sharpe-ratio-definition-formula/"

    # HTML code with image sizing
    html_str = f"""
    <a href="{link_url}" target="_blank">
        <img src="{image_url}" alt="Alt text" width="700" height="500"/>
    </a>
    """

    st.markdown(html_str, unsafe_allow_html=True)
    st.divider()

    st.info("""
             For Example: A Sharpe ratio of 1.5 indicates that the investment is generating 1.5 units of excess return for each unit of 
             risk taken, relative to the risk-free rate. It implies better risk-adjusted performance than a lower Sharpe ratio.""")
    
    stock = st.text_input("Enter Stock Symbol", "AAPL")
    if st.button("Get Sharpe Ratio & Historical Returns:"):

        # generating portfolio metrics 
        stock_sharpe = qs.utils.download_returns(stock)
        st.write("Sharpe Ratio:")
        st.write(qs.stats.sharpe(stock_sharpe))
        st.write("Historical Returns:")
        st.write(qs.stats.monthly_returns(stock_sharpe))
        returns = qs.utils.download_returns(stock)
        fig = qs.plots.snapshot(returns, title=f'{stock} Performance', show=False)
        st.write(fig)

    st.subheader("Create Optimal Portfolio")

    start = "2015-01-01"
    end = "2024-01-01"

    # variables for sharpe analysis 
    method_mu = "hist" 
    method_cov = "hist"
    hist = True
    model = "Classic"
    rm = "MV" 
    obj = "Sharpe"
    rf = 0
    l = 0


    options = st.multiselect("Select Stocks For Portfolio", stocks)

    if st.button("Generate Portfolio"):
        data = yf.download(options, start=start, end=end)
        returns = data["Adj Close"].pct_change().dropna()

        port = rp.Portfolio(returns = returns)
        port.assets_stats(methods_mu=method_mu, method_cov=method_cov)
        w = port.optimization(model=model, rm=rm, obj=obj, rf=rf, l=l, hist=hist)

        # saving w to session state to be accessed in other sidebar options 
        st.session_state['w'] = w
        
        st.write("Asset Weights:")
        st.write(w)

        fig, ax = plt.subplots()
        rp.plot_pie(w=w, title="Optimal Portfolio", others=0.05, cmap="tab20", ax=ax)
    
        st.pyplot(fig)

        frontier = port.efficient_frontier(model=model, rm=rm, points=50, rf=rf, hist=hist)
        fig, ax = plt.subplots()
        rp.plot_frontier(frontier, mu=port.mu, cov=port.cov, returns=returns, rm=rm, rf=rf, cmap="viridis", w=w)
        st.pyplot(fig)

    if st.button("Compare Your Portfolio to $SPY") and 'w' in st.session_state:

        w = st.session_state['w']

        weights_dict = {}

        # iterate through each row in the DataFrame
        for ticker, row in w.iterrows():

            # add the ticker and its weight to the dictionary
            weights_dict[ticker] = row['weights']
        
        rounded_weights = {ticker: round(weight, 2) for ticker, weight in weights_dict.items()}

        portfolio = rounded_weights

        # historical data for the stocks in the portfolio and SPY for benchmarking
        start_date = "2015-01-01"
        end_date = "2024-01-01"
        stock_symbols = list(portfolio.keys()) + ['SPY']
        data = yf.download(stock_symbols, start=start_date, end=end_date)['Adj Close']

        # calculate the value of each stock over time
        portfolio_values = pd.DataFrame()
        for stock, shares in portfolio.items():
            portfolio_values[stock] = data[stock] * shares

        # sum the individual stock values to get the total portfolio value
        portfolio_values['Total Value'] = portfolio_values.sum(axis=1)

        # Nnrmalize the total portfolio value and SPY to start at the same point for comparison
        normalized_portfolio = portfolio_values['Total Value'] / portfolio_values['Total Value'].iloc[0]
        normalized_spy = data['SPY'] / data['SPY'].iloc[0]

        # plot
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=normalized_portfolio.index, y=normalized_portfolio*100, mode='lines', name='My Portfolio'))
        fig.add_trace(go.Scatter(x=normalized_spy.index, y=normalized_spy*100, mode='lines', name='SPY Benchmark'))

        # update layout 
        fig.update_layout(
            title='Portfolio Performance vs SPY',
            xaxis_title='Date',
            yaxis_title='Percentage Change',
            yaxis_tickformat='%',  
            legend_title='Legend',
            template='plotly_white'
        )

        fig.show()

if choice == "Next Steps":
    st.subheader("Navigate to this [jupyter notebook](https://colab.research.google.com/drive/11NqRw2AIDS8mZ-kjCbjTc36-T_7tnR34#scrollTo=zLIBdzWjdqq3) for a more in depth look at quantitative finance in python!")


