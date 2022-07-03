import numpy as np


def moving_average(x: np.array) -> np.array:
    divisor = max(np.sum(~np.isnan(x)), 1)
    return np.convolve(np.nan_to_num(x), np.ones(7), "same") / divisor


def combine_ma(x: np.array) -> np.array:
    to_keep = np.logical_not(np.isnan(x))
    ma = moving_average(x)
    ma[to_keep] = x[to_keep]
    return ma
