import logging.handlers
from pathlib import Path
import os
import numpy as np
import pandas as pd
from scipy.stats import norm
import mplfinance as mpf
import datetime
from collections import Counter

from binance import Binance
from patterns_finder import PatternsFinder

redownload = False

log_file = 'log.txt'
log_to_file = True
log_to_stdout = True

logFormatter = logging.Formatter("[%(levelname) 5s/%(asctime)s] "
                                 "%(name)s: %(message)s")
rootLogger = logging.getLogger()
rootLogger.setLevel(logging.INFO)

if log_to_file:
    fileHandler = logging.handlers.RotatingFileHandler('log.txt',
                                                       encoding='utf-16',
                                                       maxBytes=5 * 1024 *
                                                                1024,
                                                       backupCount=5)
    fileHandler.setFormatter(logFormatter)
    rootLogger.addHandler(fileHandler)

if log_to_stdout:
    consoleHandler = logging.StreamHandler()
    consoleHandler.setFormatter(logFormatter)
    rootLogger.addHandler(consoleHandler)

API_KEY = 'I1YhnvGIXQdRgFMC275fUvWPoqsaIGnkYAVFLUm7qjXX6HYO6mr2rHjUop8RkDzq'
b = Binance(API_KEY)

symbols = ['BTCUSDT', 'ETHBTC', 'LTCBTC']
intervals = ['15m', '1h', '1d']
dfs = dict()


def redownload_data():
    for interval in intervals:
        for symbol in symbols:
            if not os.path.isfile('data/{0}/{1}_{0}.csv'.format(interval,
                                                                symbol)) \
                    or redownload:
                logging.info("Downloading {} {}...".format(symbol, interval))
                df = b.download_data(symbol, interval)
                logging.info("Downloaded {} klines of {} {}, writing "
                             "to file...".format(df.shape[0],
                                                 symbol, interval))

                Path("data/{}".format(interval)).mkdir(parents=True,
                                                       exist_ok=True)
                df.to_csv('data/{0}/{1}_{0}.csv'.format(interval, symbol))
                logging.info("{} {} writed to file".format(symbol, interval))
                dfs.setdefault(symbol, dict())[interval] = df


def load_data(symbol, interval):
    if symbol in dfs and interval in dfs[symbol]:
        return dfs[symbol][interval]
    else:
        logging.info("Reading {} {} from file...".format(symbol, interval))
        df = pd.read_csv('data/{0}/{1}_{0}.csv'.format(interval, symbol),
                         index_col=[0])
        dfs.setdefault(symbol, dict())[interval] = df
        return df


def update_data():
    for symbol in symbols:
        for interval in intervals:
            update_symbol(symbol, interval)


def update_symbol(symbol, interval):
    df = load_data(symbol, interval)
    df.reset_index(inplace=True)
    start_time = df['Open time'].iloc[-1] + 1
    df.set_index(['Open time'], inplace=True)
    logging.info("Updating {} {}...".format(symbol, interval))
    new_df = b.download_data(symbol=symbol, interval=interval,
                             start_time=start_time)
    logging.info("Updated {} klines of {} {}".format(new_df.shape[0],
                                                     symbol,
                                                     interval))
    dfs[symbol][interval] = df.append(new_df)


def get_percentile(p, mean, std):
    v = norm.ppf(0.5 - p / 2, loc=mean, scale=std)
    return v, 2 * mean - v


def get_percentiles(means: np.ndarray, stds: np.ndarray, p=0.8):
    return np.array([get_percentile(p, means[i], stds[i])
                     for i in range(len(means))])


def reformat_open_time(df: pd.DataFrame):
    df.reset_index(inplace=True)
    df['Open time'] = df['Open time'].apply(
        lambda x: datetime.datetime.fromtimestamp(x / 1000.0))
    df.set_index(['Open time'], inplace=True)
    return df


def plot_prediction(_ohlcv: pd.DataFrame, prediction: np.ndarray,
                    percentiles: np.ndarray, filename='plot.svg'):
    ohlcv = _ohlcv.copy()

    ohlcv.reset_index(inplace=True)
    times = ohlcv['Open time'].to_numpy()
    diffs = []

    for i in range(1, len(times)):
        diffs.append(times[i] - times[i - 1])

    c = Counter(diffs)
    interval = c.most_common(1)[0][0]

    pred_times = [times[-1] + interval]
    for i in range(len(prediction) - 1):
        pred_times.append(pred_times[-1] + interval)

    percentiles = np.swapaxes(percentiles, 0, 1)
    pred_array = np.stack((pred_times, prediction, percentiles[0],
                           percentiles[1]), axis=1)
    pre_pred = []
    for i in times:
        pre_pred.append([i, np.nan, np.nan, np.nan])
    pred_array = np.concatenate((pre_pred, pred_array), axis=0)

    pred_df = pd.DataFrame(pred_array, columns=['Open time', 'Prediction',
                                                'Lower percentile',
                                                'Upper percentile'])
    pred_df.set_index(['Open time'], inplace=True)

    post_empty = []
    for i in pred_times:
        post_empty.append([i, np.nan, np.nan, np.nan, np.nan, np.nan])

    post_df = pd.DataFrame(post_empty, columns=['Open time', 'Open', 'High',
                                                'Low', 'Close', 'Volume'])
    post_df.set_index(['Open time'], inplace=True)
    ohlcv.set_index(['Open time'], inplace=True)
    ohlcv = ohlcv.append(post_df)

    reformat_open_time(ohlcv)
    reformat_open_time(pred_df)

    if len(prediction) == 1:
        add_plot = mpf.make_addplot(pred_df, type='scatter')
    else:
        add_plot = mpf.make_addplot(pred_df)

    mpf.plot(ohlcv, type='candle', style='binance', addplot=add_plot,
             savefig='results/' + filename)


def predict(window=50, predict_steps=10, symbol='BTCUSDT', interval='1h'):
    update_symbol(symbol, interval)
    df = load_data(symbol, interval)
    data = df.tail(window)
    pf = PatternsFinder([df])

    prediction, std = pf.predict(data, predict_steps)
    percentiles = get_percentiles(prediction, std)

    return data, prediction, percentiles


redownload_data()

if not redownload:
    update_data()


'''df = pd.DataFrame({'Close': [6, 3, 0, 1, 2, 1, 1, 1, 1]})
df2 = pd.DataFrame({'Close': [6, 3, 0, 1, 3, 0, 1, 1, 2]})
pf = PatternsFinder([df, df2])

data = pd.DataFrame({'Close': [1, 3, 0, 1, 1]})
avg, std = pf.predict(data, 1)

get_percentiles(avg, std)'''

'''df = load_data('BTCUSDT', '1h')
data = df[-100:]
df = df[-200:]
pf = PatternsFinder([df])

prediction, std = pf.predict(data, 10)
percentiles = get_percentiles(prediction, std)

# prediction = np.array([60000, 60000, 60000])
# percentiles = np.array([[59000, 61000], [59000, 61000], [59000, 61000]])

plot_prediction(data, prediction, percentiles)'''
