import pandas as pd
import matplotlib.pyplot as plt
import datetime
from datahub import get_ticker_price
import plotly.graph_objs as go
from plotly.offline import plot
import statsmodels.api as sm

# Ignore warnings
import warnings
warnings.filterwarnings("ignore")

#plot_results
def plot_predictions(df_result, df_seq2seq, ticker):
    data = []

    value = go.Scatter(
        x=df_result.index,
        y=df_result.value,
        mode="lines",
        name="values",
        marker=dict(),
        text=df_result.index,
        line=dict(color="rgba(0,0,0, 0.3)"),
    )
    data.append(value)

    prediction_seq2seq = go.Scatter(
        x=df_seq2seq.index,
        y=df_seq2seq.predictions_seq2seq,
        mode="lines",
        line={"dash": "dot"},
        name='predictions seq2seq',
        marker=dict(),
        text=df_seq2seq.index,
        opacity=0.8,
    )
    data.append(prediction_seq2seq)


    prediction_dilate = go.Scatter(
        x=df_result.index,
        y=df_result.prediction_dilate,
        mode="lines",
        line={"dash": "dot"},
        name='predictions dilate',
        marker=dict(),
        text=df_result.index,
        opacity=0.8,
    )
    data.append(prediction_dilate)

    prediction_MSE = go.Scatter(
        x=df_result.index,
        y=df_result.prediction_MSE,
        mode="lines",
        line={"dash": "dot"},
        name='predictions MSE',
        marker=dict(),
        text=df_result.index,
        opacity=0.8,
    )
    data.append(prediction_MSE)


    prediction_baseline = go.Scatter(
        x=df_result.index,
        y=df_result.prediction_FCNN,
        mode="lines",
        line={"dash": "dot"},
        name='predictions FCNN',
        marker=dict(),
        text=df_result.index,
        opacity=0.8,
    )
    data.append(prediction_baseline)

    layout = dict(
        title="Predictions vs Actual Values for the {} dataset".format(ticker),
        xaxis=dict(title="Time", ticklen=5, zeroline=False),
        yaxis=dict(title="Value", ticklen=5, zeroline=False),
    )

    fig = dict(data=data, layout=layout)

    plot(fig)


# plot data
def plot_dataset(df, title):
    data = []
    value = go.Scatter(
        x=df.index,
        y=df.value,
        mode="lines",
        name="values",
        marker=dict(),
        text=df.index,
        line=dict(color="rgba(0,0,0, 0.3)"),
    )
    data.append(value)

    layout = dict(
        title=title,
        xaxis=dict(title="Date", ticklen=5, zeroline=False),
        yaxis=dict(title="Value", ticklen=5, zeroline=False),
    )

    fig = dict(data=data, layout=layout)
    plot(fig)

#plot corr
def plot_corr(df):
    f = plt.figure(figsize=(9, 9))
    plt.matshow(df.corr(), fignum=f.number)
    plt.xticks(range(df.select_dtypes(['number']).shape[1]), df.select_dtypes(['number']).columns, fontsize=14,
               rotation=45)
    plt.yticks(range(df.select_dtypes(['number']).shape[1]), df.select_dtypes(['number']).columns, fontsize=14)
    cb = plt.colorbar()
    cb.ax.tick_params(labelsize=14)
    plt.title('Correlation Matrix', fontsize=16);
    plt.show()
    return

def add_trend_cycle(df):
    cycle, trend = sm.tsa.filters.hpfilter(df.value, lamb=129600)
    df['cycle'] = cycle
    df['trend'] = trend
    return df

# plot data
def plot_trend_cycle(df, title):
    df = add_trend_cycle(df)
    data = []
    trend = go.Scatter(
        x=df.index,
        y=df.trend,
        mode="lines",
        name="Trend",
        marker=dict(),
        text=df.index,
        line=dict(color="rgba(0,0,0, 0.3)"),
    )
    data.append(trend)

    cycle = go.Scatter(
        x=df.index,
        y=df.cycle,
        mode="lines",
        name="Cycle",
        marker=dict(),
        text=df.index,
        line=dict(color="rgba(0,0,0, 0.3)"),
    )
    data.append(cycle)

    layout = dict(
        title=title,
        xaxis=dict(title="Date", ticklen=5, zeroline=False),
        yaxis=dict(title="Value", ticklen=5, zeroline=False),
    )

    fig = dict(data=data, layout=layout)
    plot(fig)

