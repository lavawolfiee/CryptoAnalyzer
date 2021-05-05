import math

import numpy as np


class PatternsFinder:
    def __init__(self, dfs):
        self.dfs = dfs

    def zscore(self, arr: np.ndarray):
        mean = arr.mean()
        std = arr.std()
        if std == 0:
            return arr, mean, std
        else:
            return (arr - mean) / std, mean, std

    def mse(self, a: np.ndarray, b: np.ndarray):
        return np.square(a - b).mean(axis=0)

    def weighted_avg_and_std(self, values: np.ndarray, weights: np.ndarray):
        average = np.average(values, weights=weights, axis=0)
        variance = np.average(np.square(values - average), weights=weights,
                              axis=0)
        return average, np.sqrt(variance)

    def predict(self, data, predict_steps):
        k = 1
        window = data.shape[0]
        weights = np.empty((0,))
        values = np.empty((0, predict_steps))
        current_prices = np.array(data.loc[:, 'Close'])
        current_prices, current_mean, current_std = self.zscore(current_prices)

        for df in self.dfs:
            prices = np.array(df.loc[:, 'Close'])

            for i in range(prices.shape[0] - window - predict_steps + 1):
                w = prices[i:i + window]
                p = prices[i + window:i + window + predict_steps]

                w, mean, std = self.zscore(w)
                mse = self.mse(w, current_prices)
                weight = 10 * math.tanh(1/(1 * (mse + 0.0000001)))
                weights = np.append(weights, weight)
                p = (p - mean) / std
                p = p * current_std + current_mean
                values = np.append(values, [p], axis=0)

        avg, std = self.weighted_avg_and_std(values, weights)
        return avg, std
