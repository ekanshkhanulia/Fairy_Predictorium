import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn # PyTorch neural network building blocks

import preprocess  # our data processing: rolling mean, differencing, normalization

CURRENT_DIR=os.path.dirname(os.path.abspath(__file__)) # folder where train.py 
# datasets/ is at project root (next to solution/), not inside solution/ — so we use dirname(CURRENT_DIR) to go up to project root, then into datasets/train.parquet
TRAIN_PATH = os.path.join(os.path.dirname(CURRENT_DIR), "datasets", "train.parquet")
VALID_PATH = os.path.join(os.path.dirname(CURRENT_DIR), "datasets", "valid.parquet")

# After preprocess feature engineering: 32 raw + 32 roll_mean + 32 roll_std + 1 temporal = 97.
FEATURES = preprocess.ENGINEERED_FEATURES
CONTEXT_LEN = 100

# LSTMPredictor input (batch, 100, 97) -> output (batch, 2). solution.py loads weights and uses same preprocess.
CHECKPOINT_PATH = os.path.join(CURRENT_DIR, "lstm_checkpoint.pth")  # save each epoch so you can resume
NUM_EPOCHS = 10

class LSTMPredictor(nn.Module):
    def __init__(self, input_size=FEATURES, hidden_size=64, num_layers=2):
        super().__init__()
        # 2 layers: first captures short-term structure, second captures longer patterns.
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=0.2,  # between LSTM layers (only when num_layers > 1)
        )
        self.dropout = nn.Dropout(0.2)  # before fc to reduce overfitting
        self.fc = nn.Linear(hidden_size, 2)  # last layer's hidden state -> (t0, t1)

    def forward(self, x):
        _, (h_n, _) = self.lstm(x)
        h = self.dropout(h_n[-1])
        return self.fc(h)



# build_xy_chunked: same logic as build_xy but yields (X, y) in chunks to avoid loading ~115 GB into RAM.
CHUNK_SIZE = 50_000  # samples per chunk (~0.6 GB per chunk)

def build_xy_chunked(path, mean=None, std=None, rolling_window=None):
    # Load the full parquet table (all sequences).
    df = pd.read_parquet(path)
    # Collect (X, y) samples until we have CHUNK_SIZE, then yield.
    X_list, y_list = [], []
    # If we have preprocessing params, we will transform each window before appending.
    use_preprocess = mean is not None and std is not None and rolling_window is not None

    # Each seq_ix is one independent sequence so that is why we are grouping by column.
    for seq_ix, grp in df.groupby(df.columns[0]):
        # Order rows by step_in_seq so time runs 0, 1, 2, ... within the sequence.
        grp = grp.sort_values(grp.columns[1])
        # states: (length_of_seq, 32) — the 32 features at each step.
        states = grp.iloc[:, 3:35].values.astype(np.float32)
        # targets: (length_of_seq, 2) — (t0, t1) at each step.
        targets = grp.iloc[:, 35:37].values.astype(np.float32)
        # need_pred: True where we must produce a prediction (e.g. steps 99–999).
        need_pred = grp.iloc[:, 2].values
        # At step t(partivlur row) we need at least 100 steps before it; so t runs from 99 to end.
        for t in range(CONTEXT_LEN - 1, len(grp)):
            # Skip steps where we don't need a prediction (e.g. warm-up).
            if not need_pred[t]:
                continue
            # X = last 100 steps of the 32 features ending at t: steps t-99 .. t (shape 100×32).
            window = states[t - CONTEXT_LEN + 1 : t + 1]
            if use_preprocess:
                window = preprocess.transform_window(window, mean, std, rolling_window)
            X_list.append(window)
            # y = targets at time t: (t0, t1) for this step.
            y_list.append(targets[t])
            # Once we have CHUNK_SIZE samples, yield one chunk and clear to save RAM.
            if len(X_list) >= CHUNK_SIZE:
                yield np.stack(X_list), np.stack(y_list)
                X_list, y_list = [], []
    # Yield any remaining samples (last chunk may have fewer than CHUNK_SIZE).
    if X_list:
        yield np.stack(X_list), np.stack(y_list)


def evaluate_valid_mse(model, path, mean, std, rolling_window):
    """Compute mean MSE on a parquet (e.g. valid.parquet). Used by tune.py to pick best hyperparams."""
    model.eval()
    total_mse = 0.0
    cnt = 0
    with torch.no_grad():
        for X_chunk, y_chunk in build_xy_chunked(path, mean=mean, std=std, rolling_window=rolling_window):
            X = torch.from_numpy(X_chunk)
            y = torch.from_numpy(y_chunk)
            pred = model(X)
            mse = torch.nn.functional.mse_loss(pred, y, reduction="sum")
            total_mse += mse.item()
            cnt += len(X)
    model.train()
    return total_mse / cnt if cnt else 0.0


def run_training(mean, std, hidden_size, lr, batch_size, num_epochs, run_checkpoint_path=None):
    """Train one model with given hyperparams. Used by tune.py. If run_checkpoint_path is set,
    save after each epoch and resume from that file if present (so you can resume a run after restart)."""
    start_epoch = 0
    model = LSTMPredictor(hidden_size=hidden_size)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    if run_checkpoint_path and os.path.exists(run_checkpoint_path):
        ckpt = torch.load(run_checkpoint_path, map_location="cpu", weights_only=False)
        model.load_state_dict(ckpt["model"])
        opt.load_state_dict(ckpt["optimizer"])
        start_epoch = ckpt["epoch"] + 1
        print(f"  resuming this run from epoch {start_epoch + 1}/{num_epochs}")
    model.train()
    for epoch in range(start_epoch, num_epochs):
        loss_sum = 0.0
        cnt = 0
        for X_chunk, y_chunk in build_xy_chunked(
            TRAIN_PATH, mean=mean, std=std, rolling_window=preprocess.ROLLING_WINDOW
        ):
            X = torch.from_numpy(X_chunk)
            y = torch.from_numpy(y_chunk)
            n = len(X)
            perm = torch.randperm(n)
            for i in range(0, n, batch_size):
                idx = perm[i : i + batch_size]
                pred = model(X[idx])
                loss = torch.nn.functional.mse_loss(pred, y[idx])
                opt.zero_grad()
                loss.backward()
                opt.step()
                loss_sum += loss.item()
                cnt += 1
        avg = loss_sum / cnt if cnt else 0.0
        print(f"  epoch {epoch+1}/{num_epochs}, train loss: {avg:.6f}")
        if run_checkpoint_path:
            torch.save({"model": model.state_dict(), "optimizer": opt.state_dict(), "epoch": epoch}, run_checkpoint_path)
    return model


def main():
    if not os.path.exists(TRAIN_PATH):
        print(f"train file not found: {TRAIN_PATH}")
        return

    # Fit preprocessing on training data and save params so solution.py can use the same pipeline.
    print("Fitting preprocessing (mean, std) on training data...")
    mean, std = preprocess.fit_normalization_from_parquet(TRAIN_PATH)
    scale_path = os.path.join(CURRENT_DIR, "lstm_scale.npz")
    preprocess.save_scale_params(scale_path, mean, std, preprocess.ROLLING_WINDOW)
    print(f"Saved scale params to {scale_path}")

    model = LSTMPredictor()
    opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
    batch_size = 256
    start_epoch = 0

    if os.path.exists(CHECKPOINT_PATH):
        print(f"Resuming from {CHECKPOINT_PATH}")
        ckpt = torch.load(CHECKPOINT_PATH, map_location="cpu")
        model.load_state_dict(ckpt["model"])
        opt.load_state_dict(ckpt["optimizer"])
        start_epoch = ckpt["epoch"] + 1
        print(f"Resuming from epoch {start_epoch + 1}/{NUM_EPOCHS}")
    else:
        print("No checkpoint found, starting from epoch 1.")

    model.train()
    for epoch in range(start_epoch, NUM_EPOCHS):
        loss_sum = 0.0
        cnt = 0
        for X_chunk, y_chunk in build_xy_chunked(
            TRAIN_PATH, mean=mean, std=std, rolling_window=preprocess.ROLLING_WINDOW
        ):
            X = torch.from_numpy(X_chunk)
            y = torch.from_numpy(y_chunk)
            n = len(X)
            perm = torch.randperm(n)
            for i in range(0, n, batch_size):
                idx = perm[i : i + batch_size]
                pred = model(X[idx])
                loss = torch.nn.functional.mse_loss(pred, y[idx])
                opt.zero_grad()
                loss.backward()
                opt.step()
                loss_sum += loss.item()
                cnt += 1
        avg_loss = loss_sum / cnt if cnt else 0.0
        print(f"epoch {epoch+1}, loss: {avg_loss:.6f}")
        torch.save(
            {"model": model.state_dict(), "optimizer": opt.state_dict(), "epoch": epoch},
            CHECKPOINT_PATH,
        )
        print(f"  checkpoint saved (epoch {epoch+1})")

    out_path = os.path.join(CURRENT_DIR, "lstm_weights.pth")
    torch.save(model.state_dict(), out_path)
    print(f"weights saved to {out_path}")


if __name__=="__main__":
    main()
    
