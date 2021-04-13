import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import csv

import datetime

import torch
import torch.nn as nn
from torch.autograd import Variable 
import pdb

import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler, MinMaxScaler

#start = (2000, 1, 1)  # 2020년 01년 01월 
#start = datetime.datetime(*start)  
#end = datetime.date.today()  # 현재 

# yahoo 에서 삼성 전자 불러오기 
#df = pdr.DataReader('005930.KS', 'yahoo', start, end)

#X = df.drop(columns='Volume')
#y = df.iloc[:, 5:6]

#print(X)
#print(y)



mm = MinMaxScaler()
ss = StandardScaler()

data_path = './data'
train_x = pd.read_csv(data_path  + "/train_x_df.csv")
train_y = pd.read_csv(data_path  + "/train_y_df.csv")
test_x = pd.read_csv(data_path  + "/test_x_df.csv")
#pdb.set_trace()
#y = train_x_df.loc[:, "sample_id":"close"] 이거 된다!
#y = train_x_df.loc[:, "sample_id":"close"]

#############################################

ma5 = train_x['close'].rolling(window=5).mean()
ma20 = train_x['close'].rolling(window=20).mean()
ma60 = train_x['close'].rolling(window=60).mean()
ma120 = train_x['close'].rolling(window=120).mean()

train_x.insert(len(train_x.columns), "MA5", ma5)
train_x.insert(len(train_x.columns), "MA20", ma20)
train_x.insert(len(train_x.columns), "MA60", ma60)
train_x.insert(len(train_x.columns), "MA120", ma120)


ma5 = train_y['close'].rolling(window=5).mean()
ma20 = train_y['close'].rolling(window=20).mean()
ma60 = train_y['close'].rolling(window=60).mean()
ma120 = train_y['close'].rolling(window=120).mean()

train_y.insert(len(train_y.columns), "MA5", ma5)
train_y.insert(len(train_y.columns), "MA20", ma20)
train_y.insert(len(train_y.columns), "MA60", ma60)
train_y.insert(len(train_y.columns), "MA120", ma120)

train_x_process = [0 for i in range(7362)]
train_y_process = [0 for i in range(7362)]


for k in range(7362):    
    train_x_process[k] = train_x.loc[train_x['sample_id']==k]
    train_y_process[k] = train_y.loc[train_y['sample_id']==k]



train_x_tensor = torch.tensor(train_x_process)
train_y_tensor = torch.tensor(train_y_process)

print("Training Shape", len(train_x_process), len(train_x_process[0]))

print("Tensor Training Shape", train_x_process.shape, train_y_process.shape)

#print("Testing Shape", X_test.shape) 

"""
torch Variable에는 3개의 형태가 있다. 
data, grad, grad_fn 한 번 구글에 찾아서 공부해보길 바랍니다. 
"""


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")  # device
print(torch.cuda.get_device_name(0))



 
class LSTM1(nn.Module):
  def __init__(self, num_classes, input_size, hidden_size, num_layers, seq_length):
    super(LSTM1, self).__init__()
    self.num_classes = num_classes #number of classes
    self.num_layers = num_layers #number of layers
    self.input_size = input_size #input size
    self.hidden_size = hidden_size #hidden state
    self.seq_length = seq_length #sequence length
 
    self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size,
                      num_layers=num_layers, batch_first=True) #lstm
    self.fc_1 =  nn.Linear(hidden_size, 128) #fully connected 1
    self.fc = nn.Linear(128, num_classes) #fully connected last layer

    self.relu = nn.ReLU() 

  def forward(self,x):
    h_0 = Variable(torch.zeros(self.num_layers, x.size(0), self.hidden_size)).to(device) #hidden state
    c_0 = Variable(torch.zeros(self.num_layers, x.size(0), self.hidden_size)).to(device) #internal state   
    # Propagate input through LSTM

    output, (hn, cn) = self.lstm(x, (h_0, c_0)) #lstm with input, hidden, and internal state
   
    hn = hn.view(-1, self.hidden_size) #reshaping the data for Dense layer next
    out = self.relu(hn)
    out = self.fc_1(out) #first Dense
    out = self.relu(out) #relu
    out = self.fc(out) #Final Output
   
    return out 


##########################Hyper Parameters################################################
num_epochs = 1000 #1000 epochs
learning_rate = 0.0001 #0.001 lr

input_size = 16 #number of features
hidden_size = 30 #number of features in hidden state
num_layers = 5 #number of stacked lstm layers

num_classes = 16 #number of output classes 

lstm1 = LSTM1(num_classes, input_size, hidden_size, num_layers, X_train_tensors_final.shape[1]).to(device)

loss_function = torch.nn.MSELoss()    # mean-squared error for regression
optimizer = torch.optim.Adam(lstm1.parameters(), lr=learning_rate)  # adam optimizer


for epoch in range(num_epochs):
  outputs = lstm1.forward(X_train_tensors_final.to(device)) #forward pass
  optimizer.zero_grad() #caluclate the gradient, manually setting to 0
 
  # obtain the loss function
  loss = loss_function(outputs, y_train_tensors.to(device))

  loss.backward() #calculates the loss of the loss function
 
  optimizer.step() #improve from loss, i.e backprop
  if epoch % 100 == 0:
    print("Epoch: %d, loss: %1.5f" % (epoch, loss.item())) 

df_X_ss = ss.transform(df.drop(columns='Volume'))
df_y_mm = mm.transform(df.iloc[:, 5:6])

df_X_ss = Variable(torch.Tensor(df_X_ss)) #converting to Tensors
df_y_mm = Variable(torch.Tensor(df_y_mm))
#reshaping the dataset
#pdb.set_trace()
df_X_ss = torch.reshape(df_X_ss, (df_X_ss.shape[0], 1, df_X_ss.shape[1]))
train_predict = lstm1(df_X_ss.to(device))#forward pass
data_predict = train_predict.data.detach().cpu().numpy() #numpy conversion
dataY_plot = df_y_mm.data.numpy()

#train_predict.to_csv('./samsung_predict.csv', index = False)
#dataY_plot.to_csv('./samsung.csv', index = False)
data_predict = mm.inverse_transform(data_predict) #reverse transformation
dataY_plot = mm.inverse_transform(dataY_plot)
rank = [i for i in range(5300)]
with open('samsung.csv', 'w', newline='\n') as f: 
    writer = csv.writer(f)
    writer.writerow(rank)
    writer.writerow(dataY_plot)
with open('samsung_predict.csv', 'w', newline='\n') as f: 
    writer = csv.writer(f)
    writer.writerow(rank)
    writer.writerow(data_predict)
print(data_predict)

