import yfinance as yf
import streamlit as st
import datetime 
import plotly.express as px
from pycaret.regression import *
import matplotlib.pyplot as plt
import os

# global function to create datasets based on user inputs 
def create_dataset(stock, start_date, end_date):
    stock_list = [stock]
    data = yf.download(tickers=stock_list, start=start_date, end=end_date)
    data = data.drop('Adj Close', axis=1)
    data['Ticker'] = stock
    data = data.dropna()
    return data

# cache the conversion to prevent computation on every rerun
@st.cache_data
def convert_df(df):
    return df.to_csv().encode('utf-8')

with st.sidebar:
    choice = st.radio('Select one:', [
        'Tutorial', 
        'Data Selection', 
        'Data Visualization', 
        'Model Selection', 
        'Download Model' 
    ])

if choice == "Tutorial":
    with st.expander("Automated Model Selection using Pycaret", expanded=False):
        st.subheader("Steps:")
        st.write("*Data Selection*")
        st.write("- Select your market data for analysis")
        st.write("*Data Visualization*")
        st.write("- Visualize your data to identify a good target column")
        st.write("*Model Selection*")
        st.write("- Run various machine learning algorithims to determine the best predictor for your selected data")
        st.write("*Download Model*")
        st.write("- Download your most accurate model for future analysis")

    st.subheader("Created by Jonathan Hofmann ---> [github](https://github.com/hofmannj0n) | [portfolio](https://www.jhofmann.me/)")
    st.write("")
    st.write("Tutorial:")

if choice == "Data Selection":

    # data selection variables 
    stocks = ["AAPL", "GOOGL", "IBM", "MSFT", "VIX", "VOO", "QQQ", "TSLA", "JPM", "AMZN", "VZ", "NVDA", "BAC", "SBUX", "NKE", "MA", "PLTR"]    
    min_timeframe = datetime.datetime.fromisoformat("2015-01-01")
    max_timeframe = datetime.datetime.fromisoformat("2023-11-21")

    # widgets to get user input
    st.header('Data Selection', divider='rainbow')
    st.write("- Use selectbox and sliders to create your data")
    st.write("- Once satisfied with parameters - select \"Generate Data\" button")
    st.write("#")
    stock = st.selectbox("Select a security for analysis", stocks, key="stock")
    st.write("#")
    start_date = st.slider("Start date", min_value=min_timeframe, max_value=max_timeframe, key="start_date")
    st.write("#")
    end_date = st.slider("End date", min_value=start_date, max_value=max_timeframe, key="end_date")
    st.write("#")

    if st.button("Generate Data"):

        # create dataset 
        data = create_dataset(stock, st.session_state.start_date, st.session_state.end_date)

        # save the generated data in st.session_state to be accessed in other sidebar options
        st.session_state.data = data
        st.success("Data generated and ready for analysis, navigate to Data Visualization in the sidebar")

        # show a preview of generated data 
        st.write("data preview:")
        st.dataframe(data.head(20))

if choice == "Data Visualization":

    # check if data is available in session state
    if "data" in st.session_state:
        data = st.session_state.data

        st.header('Data Visualizations', divider='rainbow')
        chosen_target = st.selectbox('Choose column to plot', data.columns[:5]) 

        if st.button("Plot"):
            df = st.session_state.data.copy() # make a copy to not mess with original dataset 
            window_size = 30  # 30-day moving avearge window

            # moving average
            df['Moving_Avg'] = df[chosen_target].rolling(window=window_size).mean()

            # plot
            fig = px.line(
                df,
                y=chosen_target,
                title=f"{chosen_target} Line Chart",
                color_discrete_sequence=["#9EE6CF"],
            )

            fig.add_scatter(x=df.index, y=df['Moving_Avg'], mode='lines', name='30-day Moving Avg')
            st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning("No data generated yet. Please select 'Data Selection' and generate data first.")

if choice == "Model Selection":

    if "data" in st.session_state:

        data = st.session_state.data
        chosen_target = st.selectbox('Choose the target column', data.columns[:5])

        if st.button('Run Modelling'): 

            # initializing the model
            setup(data, target=chosen_target, session_id = 123)
            exp = RegressionExperiment()
            exp.setup(data, target = chosen_target, session_id = 123)
            best_model = compare_models()
            
            # display a model summary
            st.write("Model Accuracy: (Ranked Most Accurate - to Least Accurate)")
            compare_df = pull()
            st.dataframe(compare_df.head(20))

            # display model plot 
            img = plot_model(
                best_model, plot="error", display_format="streamlit", save=True
            )
            st.image(img)
                            
            # model predictions on new data 
            st.write("Model Predictions on New Data: (Using Most Accurate Model)")
            stock_predict = predict_model(best_model)
            new_data = data.copy()
            new_data.drop(chosen_target, axis=1, inplace=True)
            predictions = predict_model(best_model, data = new_data)
            st.dataframe(predictions.head())

            # save best model to session_state 
            st.session_state.best_model = best_model

    else:
        st.warning("No data generated yet. Please select 'Data Selection' and generate data first.")

if choice == "Download Model":

    if "best_model" in st.session_state:
        
        save_model(st.session_state.best_model, 'my_best_model')

        # path to the model file
        model_path = 'my_best_model.pkl'

        # checking if the file exists
        if os.path.exists(model_path):
            with open(model_path, "rb") as fp:
                btn = st.download_button(
                    label="Download Model",
                    data=fp,
                    file_name="best_model.pkl",
                    mime="application/octet-stream"
                )
        else:
            st.error("Model file not found")

        st.markdown(":blue[Next Steps:]")
        st.write("Check out this [jupyter notebook](https://colab.research.google.com/drive/11NqRw2AIDS8mZ-kjCbjTc36-T_7tnR34#scrollTo=zLIBdzWjdqq3) for a more in depth look at quantitative finance in Python.")
        st.write("")
        st.write("Loading your best model for further analysis:")

        st.code("""
        from pycaret.utils import load_model
        from pycaret.regression import *
        
        # Load the model you generated using this app
        model = load_model('my_best_model')
        
        predictions = model.predict(new_data)
        print(predictions) """, language ="python") 
    else:
        st.warning("No model detected, navigate to Model Selection")