import numpy as np
import pandas as pd


def _to_numpy(data):
    if isinstance(data, (pd.DataFrame, pd.Series)):
        return data.values
    return data


def apply_mcar(X, y, c=0.1, random_state=42):
    rng = np.random.RandomState(random_state)
    y_arr = _to_numpy(y)
    n = len(y_arr)

    # Randomly select indices to set as missing
    S = rng.binomial(n=1, p=c, size=n)
    y_obs = np.where(S == 1, -1, y_arr)
    return y_obs


def apply_mar1(X, y, feature_idx=0, random_state=42):
    rng = np.random.RandomState(random_state)
    X_arr = _to_numpy(X)
    y_arr = _to_numpy(y)

    n = len(y_arr)

    # select feature values for the specified feature index
    x_feature = X_arr[:, feature_idx]

    x_std = (x_feature - np.mean(x_feature)) / (
        np.std(x_feature) + 1e-8
    )  # Standarization
    probs = 1 / (1 + np.exp(-x_std))  # Sigmoid to get probabilities

    S = rng.binomial(n=1, p=probs, size=n)
    y_obs = np.where(S == 1, -1, y_arr)
    return y_obs


def apply_mar2(X, y, random_state=42):
    rng = np.random.RandomState(random_state)
    X_arr = _to_numpy(X)
    y_arr = _to_numpy(y)
    n = len(y_arr)

    # Depends on every column - sum them
    scores = np.sum(X_arr, axis=1) / np.sqrt(X_arr.shape[1])
    probs = 1 / (1 + np.exp(-scores))

    S = rng.binomial(n=1, p=probs, size=n)
    y_obs = np.where(S == 1, -1, y_arr)
    return y_obs


def apply_mnar(X, y, random_state=42):
    rng = np.random.RandomState(random_state)
    X_arr = _to_numpy(X)
    y_arr = _to_numpy(y)
    n = len(y_arr)

    # Depends on X as well as the true label Y
    scores = np.sum(X_arr, axis=1) / np.sqrt(X_arr.shape[1])
    # If y=1, increase the chance of missing data, if y=0, decrease it
    combined_scores = scores + (y_arr * 2 - 1)
    probs = 1 / (1 + np.exp(-combined_scores))

    S = rng.binomial(n=1, p=probs, size=n)
    y_obs = np.where(S == 1, -1, y_arr)
    return y_obs
