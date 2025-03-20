import streamlit as st

import pandas as pd
from pandas import IndexSlice as ix
import numpy as np
from yaml import safe_load
import datetime

import yfinance as yf

import plotly.io as pio
import plotly.express as px

st.title('Portfolio visualisation')

@st.cache_data
def load_transactions():
    with open("portfolio.yaml", "rb") as f:
        portfolio = safe_load(f)
    return portfolio

# Transform dictionary into a dataframe of transaction records
@st.cache_data
def format_transactions(portfolio):
    records = []
    for ticker, transactions in portfolio.items():
        for date, details in transactions.items():
            qte = next((item['QTE'] for item in details if 'QTE' in item), None)
            price = next((pd.to_numeric(item['PRICE'], errors="coerce") for item in details if 'PRICE' in item), None)
            buy_price_yahoo = yf.download(ticker, start=date, end=date + datetime.timedelta(1), group_by="ticker")[ticker]["Close"]

            if not buy_price_yahoo.empty:
                records.append({
                    'date': date, 'ticker': ticker, 'quantity_flow': qte, 'price': price, 'buy_price_yahoo': buy_price_yahoo.values[0],
                })
            else:
                records.append({
                    'date': date, 'ticker': ticker, 'quantity_flow': qte, 'price': price, 'buy_price_yahoo': np.nan,
                })

    # Convert to DataFrame
    df = pd.DataFrame(records)

    # Sort by Date
    df.sort_values(by=["ticker", "date"], inplace=True)

    # Reset index
    df.reset_index(drop=True, inplace=True)
    df.loc[df.price.isna(), "price"] = df.loc[df.price.isna(), "buy_price_yahoo"]

    df["quantity_stock"] = df.groupby("ticker").quantity_flow.cumsum()
    df["spending_flow"] = df["quantity_flow"] * df["price"]
    df["spending_stock"] = df.groupby("ticker").spending_flow.cumsum()
    df["invested_cash_flow"] = df["quantity_flow"].where(df["quantity_flow"] > 0, 0) * df["price"]
    df["invested_cash_stock"] = df.groupby("ticker").invested_cash_flow.cumsum()

    df.drop(columns=["buy_price_yahoo"], inplace=True)

    return df


@st.cache_data
def fetch_hist_data(df, portfolio):
    # Downloading historical stock prices and merging with transaction records
    start_date = "2020-01-01"
    hist_data = yf.download(
        list(portfolio.keys()), start=start_date, 
        group_by="ticker"
    )
    hist_data = hist_data.stack(level=0, future_stack=True)
    hist_data.columns = hist_data.columns.values
    hist_data = hist_data.reset_index()
    hist_data = hist_data[["Date", "Ticker", "Close"]]
    hist_data.columns = [col.lower() for col in hist_data.columns]


    # Set start and end date
    start_date = pd.Timestamp("2020-01-01")
    end_date = datetime.datetime.today()

    # Create a complete date range
    all_dates = pd.date_range(start=start_date, end=end_date, freq="D")

    # Create a MultiIndex with all tickers and all dates
    multi_index = pd.MultiIndex.from_product([df["ticker"].unique(), all_dates], names=["ticker", "date"])

    # Reindex the DataFrame to expand the dates for all tickers
    full_df = df[["ticker", "date", "quantity_stock", "spending_stock", "invested_cash_stock"]]\
        .set_index(["ticker", "date"]).reindex(multi_index)

    # Forward-fill the quantity_stock column
    full_df["quantity_stock"] = full_df.groupby("ticker")["quantity_stock"].ffill()
    full_df["quantity_stock"] = full_df["quantity_stock"].fillna(0)

    # Forward-fill the spending_stock and invested_cash_stock columns
    full_df["spending_stock"] = full_df.groupby("ticker")["spending_stock"].ffill()
    full_df["invested_cash_stock"] = full_df.groupby("ticker")["invested_cash_stock"].ffill()

    # Merge with transaction records
    hist_data = hist_data.merge(
        full_df, how='left',
        on=["ticker", "date"]
    )
    hist_data = hist_data.set_index(["ticker", "date",])

    # Compute daily values of each stock
    hist_data["valuation"] = hist_data["quantity_stock"] * hist_data["close"]

    # Compute the balance (profit or loss) of each stock
    hist_data["balance"] = hist_data["valuation"] - hist_data["spending_stock"]

    # Compute profit rate of each stock with respect to investments made
    hist_data["profit_rate"] = hist_data["balance"] / hist_data["invested_cash_stock"]

    # Set to NaN spending_stock and invested_cash_stock whenever the daily prices were not retrieved
    # e.g when a stock has been delisted or integrated into another ticker
    # Otherwise, the cash spent on such stocks will be taken into account in the computation of profit
    # while the daily values are NaN, thus artificially deflating profits
    hist_data["spending_stock"] = hist_data["spending_stock"].where(~hist_data["close"].isna(), np.nan)
    hist_data["invested_cash_stock"] = hist_data["invested_cash_stock"].where(~hist_data["close"].isna(), np.nan)
    return hist_data

# Create a text element and let the reader know the data is loading.
portfolio_load_state = st.text('Loading transactions...')
# Load 10,000 rows of data into the dataframe.
portfolio = load_transactions()
portfolio_load_state.text("Transactions loaded !")

# Create a text element and let the reader know the data is being formatted.
data_format_state = st.text('Formatting transactions...')
# Load 10,000 rows of data into the dataframe.
df = format_transactions(portfolio)
data_format_state.text("Transactions formatted !")

# Create a text element and let the reader know the historical data is being fetched.
hist_data_state = st.text('Fetching historical data...')
# Load 10,000 rows of data into the dataframe.
hist_data = fetch_hist_data(df, portfolio)
hist_data_state.text("Historical data fetched !")

variable = st.selectbox(
    label="Variable of interest",
    options=["Profit rate", "Portfolio value"]
)

text_to_var_name = {
    "Profit rate": "profit_rate",
    "Portfolio value": "valuation",
}
year_range = st.slider('year', 2020, datetime.datetime.today().year, 2020)
fig = px.line(
    hist_data.loc[hist_data.index.get_level_values("date").year >= year_range].reset_index(), 
    x="date", 
    y=text_to_var_name[variable], 
    color="ticker", 
    title=f"{variable} over Time",
    labels={text_to_var_name[variable]: variable, "date": "Date"},
)
st.plotly_chart(fig)
