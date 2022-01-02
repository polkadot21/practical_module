import datapackage
import pandas as pd
import yfinance as yf
import datetime
import statsmodels.api as sm
import holidays


def get_BTC_data(features,
                 from_when):
    data_url = 'https://datahub.io/cryptocurrency/bitcoin/datapackage.json'

    # to load Data Package into storage
    package = datapackage.Package(data_url)

    # to load only tabular data
    resources = package.resources
    for resource in resources:
        if resource.tabular:
            data = pd.read_csv(resource.descriptor['path'])
    # datetime index
    data.index = pd.to_datetime(data.date)
    del data['date']

    # rename target
    data = data.rename(columns={"price(USD)": 'value'})

    # get rid of irrelevant features
    data = data[features]

    data = data[from_when:]

    return data


def get_ticker_price(ticker, from_when, to_when, features):
    data = yf.Ticker(ticker).history(start=from_when, end=to_when).rename(columns = {"Close":"value"})
    data = data[features]

    return data

def get_weather(features):
    path = '/Users/e.saurov/PycharmProjects/btc_pytorch/btc_in_pytorch/model/data/weather_features.csv'
    data = pd.read_csv(path).rename(columns = {"temp":"value"})
    data = data[data['city_name'] == 'Madrid']
    data = data[features]
    data = data.fillna(method='bfill')

    return data

def add_trend_cycle(df):
    cycle, trend = sm.tsa.filters.hpfilter(df.value, lamb=129600)
    df['cycle'] = cycle
    df['trend'] = trend
    return df


us_holidays = holidays.US()
def is_holiday(date):
    date = date.replace(hour = 0)
    return 1 if (date in us_holidays) else 0

def add_holiday_col(data, holidays):
    return data.assign(is_holiday = data.index.to_series().apply(is_holiday))

