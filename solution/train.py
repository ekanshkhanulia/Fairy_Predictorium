import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn #pytocrh neural netwrok building blaoks

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



# build_xy: load parquet, for each sequence take every row where we need a prediction; X = last 100 steps (32 features), y = (t0, t1) for that row. Returns (X, y) as numpy arrays.
def build_xy(path):#
    df=pd.read_parquet(path)
    # Collect each sample: X_list = list of 2D arrays (100×32); y_list = list of (t0, t1). Same order so X_list[i] and y_list[i] are one training example. We stack them at the end into (N, 100, 32) and (N, 2).
    X_list,y_list=[],[]

    for seq_ix,grp in df.groupby(df.columns[0]): #group by seq_ix (sequence index) .split the table so all rows withsame seq ix are togethere
        grp=grp.sort_values(grp.columns[1]) #we sort step in seq to make sure the steps are in order
        states = grp.iloc[:, 3:35].values.astype(np.float32)    # 32 features (cols 3–34)
        targets = grp.iloc[:, 35:37].values.astype(np.float32)  # t0, t1 (cols 35–36)
        need_pred = grp.iloc[:, 2].values            # need_prediction (col 2)
        for t in range(CONTEXT_LEN - 1, len(grp)):   # from step 99 to end
            if not need_pred[t]:
                continue
            # Slice last 100 steps ending at t: rows (t-99) to t inclusive → shape (100, 32). 
            window = states[t - CONTEXT_LEN + 1 : t + 1]
            X_list.append(window)
            y_list.append(targets[t])
    return np.stack(X_list), np.stack(y_list)


def main():
    if not os.path.exists(TRAIN_PATH):
        print(f"tarin file not focund: {TRAIN_PATH}")
        return 

    print("loading data...")
    X,y=build_xy(TRAIN_PATH)

    print(f"samples:{len(X)}")
    X=torch.from_numpy(X)
    y=torch.from_numpy(y)
    model=LSTMPredictor()
    opt=torch.optim.Adam(model.parameters(),lr=1e-3)
    batch_size=256
    n=len(X)
    for epoch in range(10):
        perm=torch.randperm(n)
        loss_sum=0.0
        cnt=0
        for i in range(0,n,bacth_size):
            idx=perm[i:i+bacth_size]
            pred=model(X[idx])
            loss=torch.nn.functional.mse_loss(pred,y[idx])
            opt.zero_grad()
            loss.backward()
            opt.step()
            loss_sum+=loss.item()
            cnt+=1
        print(f"epoch {epoch+1}, loss: {loss_sum/cnt:.6f}")
    out_path=os.path.join(CURRENT_DIR,"lstm_weights.pth")
    torch.save(model.state_dict(),out_path)
    print(f"weights saved to {out_path}")


if __name__=="__main__":
    main()
    
