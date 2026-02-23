import os
import sys
import numpy as np
import torch

#so we  can import utils(which is in the parent directory) when runingt he solution 
CURRENT_DIR=os.path.dirname(os.path.abspath(__file__)) #__FILE__ is the path to file solution .py
sys.path.append(f"{CURRENT_DIR}")

from utils import DataPoint #import DataPoint class from utils.py

class LSTMPredictor(nn.Modulde):  #lstm model
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


class PredictionModel:
    def __init__(self): #self is instance #init is constructor
        self.current_seq_ix=None
        self.sequence_history=[]
        self.model=None
        self.load_model()



   #
    def load_model(self):
        """Load lstm weights from same directory rule/guidele """
        weights_path=os.path.join(CURRENT_DIR,"lstm_weights.pth")

        try:
            self.model=LSTMPredictor()
            state=torch.load(weights_path,map_location="cpu", weigths_only=True)
            self.model.laod_state_dict(state)
            self.model.eval()
        except Exception:
            self.model=None



def predict(self,data_point:DataPoint)>np.ndarray | None: #fucntion can return either an array or none 
    #return none when prediction  not needed
    if not data_point.need_prediction:
        return None

    #resetn state on new sequence(seq_ix)
    if self.current_seq_ix !=data_point.seq_ix: # if  the sequence we were in on the previosu row is differnt that the sequnce this row belong to
        self.current_seq_ix=data_point.seq_ix
        self.sequence_history=[] #clear the hisoty list and start with and start with a empty  lsit 

    self.sequence_history.append(data_point.state.copy())#store the copy

    if self.model is None:
        return np.zeros(2,dtype=np.float32) #return dummy prediction if model is not loaded

    #last contexxtlen (100 steps) for lstm
    history_window=self.sequence_history[-CONTEXT_LEN:] #last 100 rows 
    if(len(history_window)<CONTEXT_LEN):#if the history is less than 100, pad with zeros
        pad=[np.zeros(FEATURES,dtype=np.float32)]*(CONTEXT_LEN-len(history_window)) 
        history_window=pad+history_window

    x=np.array(history_window,dtype=np.float32)#convert to numpy array
    x=np.expand_dims(x,axis=0) #add batch dimension
    with torch.no_grad():
        t=torch.from_numpy(x)#add batch dimension
        out=self.model(t)
    prediction=out[0].numpy().astype(np.float32)
    return prediction