import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn # PyTorch neural network building blocks

CURRENT_DIR=os.path.dirname(os.path.abspath(__file__)) # folder where train.py 
# datasets/ is at project root (next to solution/), not inside solution/ — so we use dirname(CURRENT_DIR) to go up to project root, then into datasets/train.parquet
TRAIN_PATH=os.path.join(os.path.dirname(CURRENT_DIR),"datasets","train.parquet") #path to train.parquet file

FEATURES=32 #32 features per row
CONTEXT_LEN=100 #LAST 100 STEPS OF THE SEQUENCE 	

# LSTMPredictor  Input (batch, 100, 32) -> output (batch, 2). We train it here and save weights; solution.py loads them.

class LSTMPredictor(nn.Module):
    def __init__(self,input_size=FEATURES,hidden_size=64,num_layers=1): #input_size is the number of features, hidden_size is the number of hidden units
        super().__init__()
        self.lstm=nn.LSTM(input_size=input_size,hidden_size=hidden_size,num_layers=num_layers,batch_first=True,) #lstm layer
        self.fc=nn.Linear(hidden_size,2)    #fully connected layer to output 2 values    64->2(t0 and t1)
         ## Creates the linear layer and stores it as self.fc. Runs once in __init__. Maps 64 numbers (hidden_size) -> 2 numbers (t0, t1). No input is passed here; we only define the layer.


    def forward(self,x):
        _,(h_n,_)=self.lstm(x)     
        return self.fc(h_n[-1])



# build_xy_chunked: same logic as build_xy but yields (X, y) in chunks to avoid loading ~115 GB into RAM.
CHUNK_SIZE = 50_000  # samples per chunk (~0.6 GB per chunk)

def build_xy_chunked(path):
    # Load the full parquet table (all sequences).
    df = pd.read_parquet(path)
    # Collect (X, y) samples until we have CHUNK_SIZE, then yield.
    X_list, y_list = [], []

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
            X_list.append(window)
            # y = targets at time t: (t0, t1) for this step.
            y_list.append(targets[t])
            # Once we have CHUNK_SIZE samples, yield one chunk and clear to save RAM.
            if len(X_list) >= CHUNK_SIZE:
                chunk_num += 1
                print(f"  Chunk {chunk_num}: yielding {len(X_list):,} samples", flush=True)
                yield np.stack(X_list), np.stack(y_list)
                X_list, y_list = [], []
    # Yield any remaining samples (last chunk may have fewer than CHUNK_SIZE).
    if X_list:
        chunk_num += 1
        print(f"  Chunk {chunk_num}: yielding {len(X_list):,} samples (last)", flush=True)
        yield np.stack(X_list), np.stack(y_list)


def main():
    if not os.path.exists(TRAIN_PATH):
        print(f"train file not found: {TRAIN_PATH}")
        return

    print("Loading data (chunked)...")
    model = LSTMPredictor()
    model.train()
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    batch_size = 256

    for epoch in range(10):
        loss_sum = 0.0
        cnt = 0
        for X_chunk, y_chunk in build_xy_chunked(TRAIN_PATH):
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
    out_path=os.path.join(CURRENT_DIR,"lstm_weights.pth")
    torch.save(model.state_dict(),out_path)
    print(f"weights saved to {out_path}")


if __name__=="__main__":
    main()
    
