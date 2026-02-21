import os
import sys
import numpy as np

#so we  can import utils(which is in the parent directory) when runingt he solution 
CURRENT_DIR=os.path.dirname(os.path.abspath(__file__)) #__FILE__ is the path to file solution .py
sys.path.append(f"{CURRENT_DIR}")

from utils import DataPoint #import DataPoint class from utils.py

class LSTMPredictor(nn.Modulde):
    def __init__(self,input_size=FEATURES,hidden_size=64,num_layers=1):
        super().__init__()
        self.lstm=nn.LSTM(input_size=input_size,hidden_size=hidden_size,num_layers=num_layers,batch_first=True,) #lstm layer
        # Input = last LSTM hidden state (size 64). Output = 2 numbers (t0 and t1).
        self.fc=nn.Linear(hidden_size,2)

    def forward(self,x): #calling the model #x:input data 
        #x:(bacth (no. of sequences), seq_len(no. of steps 100 context window),input_size(no. of features 32))
        _,(h_n,_)=self.lstm(x)

        #h_n:(num_layers, batch,hidden_szie)->take last layer
        out=self.fc(h_n[-1]) //
        return out