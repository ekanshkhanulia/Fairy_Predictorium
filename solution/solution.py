"""
Submission entry point. Must define PredictionModel with predict(data_point).
Uses same preprocessing and model as train.py (97 features, 2-layer LSTM).
"""
import json
import os
import sys
import numpy as np
import torch
import torch.nn as nn

# CURRENT_DIR = folder containing solution.py (same as submission zip root when submitted).
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, CURRENT_DIR)
# Utils may be in parent when run from repo.
if os.path.dirname(CURRENT_DIR) not in sys.path:
    sys.path.insert(0, os.path.dirname(CURRENT_DIR))

from utils import DataPoint

import preprocess

CONTEXT_LEN = 100
FEATURES_RAW = 32  # raw state from DataPoint; after preprocess we get 97.


class LSTMPredictor(nn.Module):
    """Must match train.py: 2 layers, dropout, input 97, output 2."""
    def __init__(self, input_size=preprocess.ENGINEERED_FEATURES, hidden_size=64, num_layers=2):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=0.2,
        )
        self.dropout = nn.Dropout(0.2)
        self.fc = nn.Linear(hidden_size, 2)

    def forward(self, x):
        _, (h_n, _) = self.lstm(x)
        h = self.dropout(h_n[-1])
        return self.fc(h)


class PredictionModel:
    """Required by competition: init + predict(data_point) -> None or np.ndarray shape (2,)."""

    def __init__(self):
        self.current_seq_ix = None
        self.sequence_history = []
        self.model = None
        self.mean = None
        self.std = None
        self.rolling_window = preprocess.ROLLING_WINDOW
        self.load_model()

    def load_model(self):
        """Load scale params (for preprocessing) and LSTM weights. Same pipeline as train.py."""
        scale_path = os.path.join(CURRENT_DIR, "lstm_scale.npz")
        weights_path = os.path.join(CURRENT_DIR, "lstm_weights.pth")
        params_path = os.path.join(CURRENT_DIR, "lstm_best_params.json")

        if os.path.exists(scale_path):
            self.mean, self.std, self.rolling_window = preprocess.load_scale_params(scale_path)
        else:
            self.mean = None
            self.std = None

        hidden_size = 64
        if os.path.exists(params_path):
            with open(params_path) as f:
                best = json.load(f)
                hidden_size = best.get("hidden_size", 64)

        try:
            self.model = LSTMPredictor(
                input_size=preprocess.ENGINEERED_FEATURES,
                hidden_size=hidden_size,
                num_layers=2,
            )
            state = torch.load(weights_path, map_location="cpu", weights_only=True)
            self.model.load_state_dict(state)
            self.model.eval()
        except Exception:
            self.model = None

    def predict(self, data_point: DataPoint) -> np.ndarray | None:
        """Return None when need_prediction is False; else return (t0, t1) as shape (2,) float32."""
        if not data_point.need_prediction:
            return None

        # Reset state on new sequence (required by rules).
        if self.current_seq_ix != data_point.seq_ix:
            self.current_seq_ix = data_point.seq_ix
            self.sequence_history = []

        self.sequence_history.append(data_point.state.copy())

        if self.model is None:
            return np.zeros(2, dtype=np.float32)

        # Build last CONTEXT_LEN steps (100 x 32); pad with zeros if shorter.
        history_window = self.sequence_history[-CONTEXT_LEN:]
        if len(history_window) < CONTEXT_LEN:
            pad = [np.zeros(FEATURES_RAW, dtype=np.float32)] * (CONTEXT_LEN - len(history_window))
            history_window = pad + history_window
        window = np.array(history_window, dtype=np.float32)

        # Same preprocessing as training: feature eng + detrend + diff + normalize.
        if self.mean is not None and self.std is not None:
            window = preprocess.transform_window(window, self.mean, self.std, self.rolling_window)

        x = torch.from_numpy(window).unsqueeze(0)
        with torch.no_grad():
            out = self.model(x)
        prediction = out[0].numpy().astype(np.float32)
        return prediction
