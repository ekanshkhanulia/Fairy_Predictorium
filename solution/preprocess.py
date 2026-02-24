"""
Data preprocessing for the LSTM model: rolling mean (detrend), differencing, normalization.

This file holds all the steps we use to clean and scale the 32 features before they go
into the LSTM. Both train.py and solution.py must use the SAME pipeline and the SAME
fitted parameters (mean, std), so we define everything here and save/load the parameters
to a .npz file.

Pipeline order for each 100×32 window:
  0. Feature engineering: add rolling mean, rolling std, and temporal (step index).
  1. Rolling mean detrend: subtract a causal rolling average to remove local trend.
  2. Differencing: use step-to-step changes to make the series more stationary.
  3. Normalization: (x - mean) / std using mean and std fitted on the training set only.

After feature engineering each window is (100, 97): 32 raw + 32 rolling_mean + 32 rolling_std + 1 temporal.
"""

import numpy as np
import pandas as pd
import os

NUM_RAW_FEATURES = 32
ENGINEERED_FEATURES = NUM_RAW_FEATURES * 3 + 1  # 32 + 32 + 32 + 1 = 97

ROLLING_WINDOW = 10


def add_engineered_features(window, rolling_window=ROLLING_WINDOW):
    """Add rolling mean, rolling std, and temporal (step index). (100, 32) -> (100, 97)."""
    T, F = window.shape
    roll_mean = np.empty((T, F), dtype=np.float32)
    roll_std = np.empty((T, F), dtype=np.float32)
    for t in range(T):
        start = max(0, t - rolling_window + 1)
        seg = window[start : t + 1]
        roll_mean[t] = np.mean(seg, axis=0)
        roll_std[t] = np.std(seg, axis=0)
    roll_std = np.maximum(roll_std, 1e-8)
    temporal = (np.arange(T, dtype=np.float32) / max(T - 1, 1)).reshape(-1, 1)
    return np.concatenate([window, roll_mean, roll_std, temporal], axis=1)


def rolling_mean_detrend(states, window):
    """
    Remove local trend by subtracting a causal rolling mean from each feature.

    For each of the 32 features we compute, at each time step t, the average of
    the last `window` steps (including t). Then we subtract that average from
    the current value. So we get "how much above or below the recent average"
    instead of the raw level, which helps with non-stationarity.

    Args:
        states: array of shape (T, 32) — T time steps, 32 features.
        window: int — number of steps to average over (e.g. 10).

    Returns:
        array of shape (T, 32) — detrended values (original minus rolling mean).
    """
    T, F = states.shape
    out = np.empty_like(states, dtype=np.float32)

    for t in range(T):
        # Causal window: we only use steps from 0 to t (no future).
        start = max(0, t - window + 1)
        # Average over steps [start, start+1, ..., t] for each of the 32 features.
        out[t] = np.mean(states[start : t + 1], axis=0)

    # Detrend = original minus rolling mean.
    return states - out


def difference(states):
    """
    First-order differencing along the time axis to make the series more stationary.

    At each time step t we replace the value with (value at t) - (value at t-1).
    The first row has no previous step, so we set it to 0 (or we could keep the
    first row as-is; here we use 0 so the shape stays (T, 32) and the first
    step is "no change").

    Args:
        states: array of shape (T, 32).

    Returns:
        array of shape (T, 32): diff[0] = 0, diff[t] = states[t] - states[t-1] for t >= 1.
    """
    T, F = states.shape
    out = np.zeros_like(states, dtype=np.float32)
    out[1:] = states[1:] - states[:-1]
    return out


def fit_normalization_from_parquet(parquet_path):
    """
    Compute mean and std for the 97 engineered features.

    We fit mean/std on the 32 raw features from the parquet, then extend to 97:
    [mean_32, mean_32, std_32, 0.5] and [std_32, std_32, std_32, ~0.29] so rolling
    stats and temporal have sensible scales.
    """
    df = pd.read_parquet(parquet_path)
    feats = df.iloc[:, 3:35].values.astype(np.float32)
    mean_32 = np.mean(feats, axis=0).astype(np.float32)
    std_32 = np.std(feats, axis=0).astype(np.float32)
    std_32 = np.maximum(std_32, 1e-8)
    # Extend to 97: raw, roll_mean, roll_std, temporal (temporal ~ uniform [0,1] -> mean 0.5, std ~0.29).
    mean_97 = np.concatenate([mean_32, mean_32, std_32, np.array([0.5], dtype=np.float32)])
    std_97 = np.concatenate([std_32, std_32, std_32, np.array([0.29], dtype=np.float32)])
    return mean_97, std_97


def normalize(states, mean, std):
    """
    Standardize the features: (states - mean) / std.

    mean and std are (32,) — one value per feature. We subtract mean and
    divide by std so that each feature has roughly zero mean and unit variance,
    which helps the LSTM train stably.

    Args:
        states: array of shape (..., 32) — can be (T, 32) or (batch, T, 32).
        mean: array of shape (32,).
        std:  array of shape (32,).

    Returns:
        array same shape as states, dtype float32.
    """
    return ((states - mean) / std).astype(np.float32)


def transform_window(window, mean, std, rolling_window=ROLLING_WINDOW):
    """
    Full pipeline: feature engineering -> detrend -> differencing -> normalization.
    Input (100, 32), output (100, 97) preprocessed for the LSTM.
    """
    window = add_engineered_features(window, rolling_window)
    detrended = rolling_mean_detrend(window, rolling_window)
    differenced = difference(detrended)
    scaled = normalize(differenced, mean, std)
    return scaled


def save_scale_params(save_path, mean, std, rolling_window):
    """
    Save the fitted preprocessing parameters to a .npz file.

    We need to save these so that solution.py (inference) can load them and
    apply the exact same normalization and rolling window when it sees new data.
    If we don't use the same mean, std, and rolling_window at inference, the
    model will get inputs in a different scale and predictions will be wrong.

    Args:
        save_path: path to the file (e.g. 'solution/lstm_scale.npz').
        mean: array of shape (97,) after feature engineering.
        std:  array of shape (97,).
        rolling_window: int (e.g. 10).
    """
    np.savez(save_path, mean=mean, std=std, rolling_window=np.int32(rolling_window))


def load_scale_params(load_path):
    """
    Load the preprocessing parameters from a .npz file (saved during training).

    Use this in solution.py when loading the model so you can preprocess each
    incoming 100×32 window the same way as in training.

    Args:
        load_path: path to the .npz file (e.g. 'solution/lstm_scale.npz').

    Returns:
        mean: array of shape (97,).
        std:  array of shape (97,).
        rolling_window: int.
    """
    data = np.load(load_path)
    mean = data["mean"].astype(np.float32)
    std = data["std"].astype(np.float32)
    rolling_window = int(data["rolling_window"])
    return mean, std, rolling_window
