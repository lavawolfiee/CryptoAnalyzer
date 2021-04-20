import requests
from requests.adapters import HTTPAdapter
from urllib3 import Retry

import logging.handlers
import time

import numpy as np
import pandas as pd


class Binance:
    BASE_URL = "https://api.binance.com"
    PRICE_TICKER_URL = "/api/v3/ticker/bookTicker"
    INFO_URL = "/api/v3/exchangeInfo"
    ORDER_URL = "/api/v3/order"
    TEST_ORDER_URL = "/api/v3/order/test"
    KLINES_URL = "/api/v3/klines"

    OK_STATUS = 200
    RATE_LIMIT_STATUS = 429

    CURRENCIES_EXPIRE_TIME = 60 * 10
    CHECKS_DELAY = 0
    PRICES_EXPIRE_TIME = CHECKS_DELAY
    MAX_BACKOFF_TIME = 60 * 30
    BACKOFF_FACTOR = 60

    def __requests_retry_session(self, retries=5,
                                 backoff_factor=0.3,
                                 status_forcelist=(500, 502, 504),
                                 ):
        session = requests.Session()
        retry = Retry(
            total=retries,
            read=retries,
            connect=retries,
            backoff_factor=backoff_factor,
            status_forcelist=status_forcelist,
        )
        adapter = HTTPAdapter(max_retries=retry)
        session.mount('http://', adapter)
        session.mount('https://', adapter)
        return session

    def __init__(self, api_key):
        self.API_KEY = api_key
        self.session = self.__requests_retry_session()

    def get_request(self, url, headers=None, params=None):
        if params is None:
            params = {}
        if headers is None:
            headers = {}

        r = self.session.get(url, timeout=10, headers=headers, params=params)

        if r.status_code == self.OK_STATUS:
            return r
        elif r.status_code == self.RATE_LIMIT_STATUS:
            logging.warning("Rate limit reached. Waiting for %s seconds" % (
                    int(r.headers['Retry-After']) + 5))
            time.sleep(int(r.headers['Retry-After']) + 5)
            return self.get_request(url, headers=headers, params=params)
        else:
            logging.error('Received bad status code: %d. Error: %s' % (
                r.status_code, r.text))
            raise Exception('Received bad status code: %d. Error: %s' % (
                r.status_code, r.text))

    def get_klines(self, symbol, interval, start_time=None, end_time=None,
                   limit=1000):
        params = {'symbol': symbol,
                  'interval': interval,
                  'limit': limit}
        if start_time:
            params['startTime'] = start_time
        if end_time:
            params['endTime'] = end_time

        r = self.get_request(self.BASE_URL + self.KLINES_URL, params=params)

        return np.array(r.json(), dtype=np.double)

    def download_data(self, symbol='BTCUSDT', interval='1h', limit=None,
                      start_time=0):
        if limit is None:
            arr = self.get_klines(symbol, interval, limit=1000)
        else:
            arr = self.get_klines(symbol, interval, limit=min(limit, 1000))

        last_time = int(arr[0][0]) - 1

        while last_time > start_time and \
                (limit is None or arr.shape[0] < limit):
            a = self.get_klines(symbol, interval, start_time=start_time,
                                end_time=last_time, limit=1000)
            arr = np.append(a, arr, 0)
            last_time = int(a[0][0]) - 1

            if a.shape[0] < 1000:
                break

        if limit is not None and arr.shape[0] > limit:
            arr = arr[-limit:]

        bool_arr = arr[:, 0] >= start_time
        arr = arr[bool_arr]
        arr = np.delete(arr, np.s_[6:], 1)
        df = pd.DataFrame(np.array(arr),
                          columns=['Open time', 'Open', 'High', 'Low', 'Close',
                                   'Volume'])
        df = df.set_index(['Open time'])
        return df
