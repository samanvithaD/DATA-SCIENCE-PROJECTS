import numpy as np, pandas as pd, streamlit as st
import matplotlib.pyplot as plt
from scipy.stats import norm
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error

st.set_page_config(page_title="Retail Forecasting", layout="wide")
st.title("Time Series Forecasting of Retail Sales")


@st.cache_data
def load():
    df = pd.read_csv("data/retail_store_inventory.csv", parse_dates=["Date"])
    return df, df.groupby("Date")["Units Sold"].sum().asfreq("D")


df, sales = load()
split = int(len(sales) * 0.8)
train, test = sales.iloc[:split], sales.iloc[split:]


@st.cache_data
def evaluate():
    fc = {"Historical mean": pd.Series(train.mean(), index=test.index),
          "Last value": pd.Series(train.iloc[-1], index=test.index),
          "ARIMA(1,0,1)": ARIMA(train, order=(1, 0, 1)).fit().forecast(len(test))}
    rows = {k: {"MAE": mean_absolute_error(test, v), "RMSE": ((test - v) ** 2).mean() ** 0.5,
                "MAPE %": mean_absolute_percentage_error(test, v) * 100} for k, v in fc.items()}
    return fc, pd.DataFrame(rows).T.round(2)


fc, table = evaluate()
t1, t2, t3 = st.tabs(["Data", "Model comparison", "Forecast and safety stock"])

with t1:
    adf_p = adfuller(sales)[1]
    st.write(f"{len(sales)} days, {sales.index.min().date()} to {sales.index.max().date()}. "
             f"Mean {sales.mean():,.0f} units a day. ADF p-value {adf_p:.4f}, so the series is "
             f"{'stationary' if adf_p < 0.05 else 'not stationary'}.")
    st.line_chart(pd.DataFrame({"Daily units": sales, "30-day mean": sales.rolling(30).mean()}))

with t2:
    st.write("Chronological split: first 80% of days train, last 20% test. Lower is better.")
    st.dataframe(table, use_container_width=True)
    st.write("Models tie with the historical mean, so daily sales are mostly noise around a constant level.")
    d = pd.DataFrame({"actual": test, "predicted": fc["ARIMA(1,0,1)"]})
    wk = d.resample("W").sum()[d.resample("W").size() == 7]
    st.metric("Daily MAPE", f"{mean_absolute_percentage_error(test, fc['ARIMA(1,0,1)']) * 100:.2f}%")
    st.metric("Weekly MAPE", f"{mean_absolute_percentage_error(wk.actual, wk.predicted) * 100:.2f}%")
    st.line_chart(pd.DataFrame({"Actual": test, "ARIMA": fc["ARIMA(1,0,1)"], "Mean": fc["Historical mean"]}))

with t3:
    steps = st.slider("Forecast horizon (days)", 7, 180, 90)
    level = st.slider("Service level", 0.80, 0.99, 0.95)
    final = ARIMA(sales, order=(1, 0, 1)).fit()
    f = final.get_forecast(steps).summary_frame(alpha=0.05)
    f.index = pd.date_range(sales.index[-1] + pd.Timedelta(days=1), periods=steps, freq="D")
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(sales.iloc[-120:], color="black", lw=1, label="History")
    ax.plot(f["mean"], color="#C44E52", label="Forecast")
    ax.fill_between(f.index, f["mean_ci_lower"], f["mean_ci_upper"], color="#C44E52", alpha=.2, label="95% interval")
    ax.legend(); st.pyplot(fig)
    err = test - fc["ARIMA(1,0,1)"]
    weekly_std = err.std() * np.sqrt(7)
    safety = norm.ppf(level) * weekly_std
    st.metric("Average weekly demand", f"{f['mean'].mean() * 7:,.0f} units")
    st.metric(f"Weekly safety stock at {level:.0%}", f"{safety:,.0f} units")
    st.caption("Safety stock = z x daily error std x sqrt(7). Assumes daily errors are independent.")
