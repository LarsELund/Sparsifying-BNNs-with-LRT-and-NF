#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import os
from sklearn.model_selection import StratifiedKFold
from torch.optim.lr_scheduler import MultiStepLR


import sys
current_dir = os.getcwd()
sys.path.append('../layers')
sys.path.append('../config')
from config import config


# select the device
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
LOADER_KWARGS = {'num_workers': 1, 'pin_memory': True} if torch.cuda.is_available() else {}



# define the parameters
BATCH_SIZE = 12240
TEST_BATCH_SIZE = 1360 # number of test samples
CLASSES = 7

TRAIN_SIZE = 12240
TEST_SIZE = 1360
NUM_BATCHES = TRAIN_SIZE/BATCH_SIZE

#labs = (Seker, Barbunya, Bombay, Cali, Dermosan, Horoz, Sira) 

TEST_SAMPLES = config['test_samples']
epochs = config['num_epochs']
num_transforms = config['num_transforms']
dim = config['hidden_dim']
lr = config['lr']




assert (TRAIN_SIZE % BATCH_SIZE) == 0
assert (TEST_SIZE % TEST_BATCH_SIZE) == 0



import pandas as pd
df = pd.read_excel('Dry_Bean_Dataset.xlsx',skiprows = 11) #jremove 11 samples

y = df.iloc[:,16]
y = y.replace('SEKER',0)
y = y.replace('BARBUNYA',1)
y = y.replace('BOMBAY',2)
y = y.replace('CALI',3)
y = y.replace('DERMASON',4)
y = y.replace('HOROZ',5)
y = y.replace('SIRA',6)
df.drop(df.columns[16], inplace=True, axis=1)
full_data = np.column_stack((df,y))


p = full_data.shape[1] - 1

class Net(nn.Module):
  def __init__(self):
    super(Net, self).__init__()
    
    self.fc1 = nn.Linear(p, dim)
    self.fc2 = nn.Linear(dim, 7)
    self.Dropout = nn.Dropout(0.9) #corresponding to 0.1 inclusion rate for LBBNNs
    
    self.act = nn.ReLU()
  
  def forward(self, x, sample=False):
      x = self.act(self.fc1(x))
      x = self.Dropout(x)
      x = F.log_softmax((self.fc2.forward(x)), dim=1)

      return x
  
    
# Stochastic Variational Inference iteration
def train(net,train_data, optimizer, batch_size = BATCH_SIZE):
    net.train()
    old_batch = 0
    for batch in range(int(np.ceil(train_data.shape[0] / batch_size))):
        batch = (batch + 1)
        _x = train_data[old_batch: batch_size * batch,0:p]
        _y = train_data[old_batch: batch_size * batch, -1]
        
        old_batch = batch_size * batch
        data = _x.to(DEVICE)
        target = _y.to(torch.long).to(DEVICE)
        net.zero_grad()
        outputs = net(data, sample=True)
        negative_log_likelihood = F.nll_loss(outputs, target, reduction='sum') 
        loss = negative_log_likelihood 
        loss.backward()
        optimizer.step()
    print('loss', loss.item())
    print('nll', negative_log_likelihood.item())
    return negative_log_likelihood.item(), loss.item()



def enable_dropout(model):
    """ Function to enable the dropout layers during test-time """
    for m in model.modules():
        if m.__class__.__name__.startswith('Dropout'):
            m.train()
            
            
def ece_score(output, label, n_bins=10):

    py = np.array(output)
    y_test = np.array(label,dtype = int)
     
    

    if y_test.ndim > 1:
        y_test = np.argmax(y_test, axis=1)
   

    py_index = np.argmax(py, axis=1)
    py_value = []
    for i in range(py.shape[0]):
        py_value.append(py[i, py_index[i]])
    py_value = np.array(py_value)
    
   
    
    acc, conf = np.zeros(n_bins), np.zeros(n_bins)
    Bm = np.zeros(n_bins)
    for m in range(n_bins):
        a, b = m / n_bins, (m + 1) / n_bins
        for i in range(py.shape[0]):
            if py_value[i] > a and py_value[i] <= b:
                Bm[m] += 1
                if py_index[i] == y_test[i]:
                    acc[m] += 1
                conf[m] += py_value[i]
        if Bm[m] != 0:
            acc[m] = acc[m] / Bm[m]
            conf[m] = conf[m] / Bm[m]
    ece = 0
    for m in range(n_bins):
        ece += Bm[m] * np.abs((acc[m] - conf[m]))
    return ece / sum(Bm)

def test_ensemble(net,test_data):
    net.eval()
    enable_dropout(net)
    metr = []
    ensemble = []
    ece = 0
    nll = nn.NLLLoss(reduction='none')
    with torch.no_grad():
        old_batch = 0
        for batch in range(int(np.ceil(test_data.shape[0] / TEST_BATCH_SIZE))):
            batch = (batch + 1)
            _x = test_data[old_batch: TEST_BATCH_SIZE * batch, 0:p]
            _y = test_data[old_batch: TEST_BATCH_SIZE * batch, -1]

            old_batch = TEST_BATCH_SIZE * batch

            data = _x.to(DEVICE)
            target = _y.to(torch.long).to(DEVICE)
            outputs = torch.zeros(TEST_SAMPLES, TEST_BATCH_SIZE, CLASSES).to(DEVICE)
            logliks = torch.zeros(TEST_SAMPLES,TEST_BATCH_SIZE)
            for i in range(TEST_SAMPLES):
                outputs[i] = net(data, sample=True)
                logliks[i] = - nll(outputs[i],target)
    
    
            output1 = outputs.mean(0)
            pred1 = output1.max(1, keepdim=True)[1]  # index of max log-probability
            
            
            tar = target.detach().cpu().numpy()
    
           
            o= output1.detach().cpu().numpy()
       
        
            ece  = ece_score(np.exp(o),tar)
            
            
            ##for full model averaging
            likelihoods = torch.exp(logliks)
            first_term = torch.log(likelihoods.mean(axis = 0))
            second_term = logliks.mean(axis = 0)
            
            waic = 2 * torch.sum(first_term - second_term)
            
            var = logliks.var(axis = 0).sum()
        


         
            b = pred1.eq(target.view_as(pred1)).sum().item()
            ensemble.append(b)


    metr.append(np.sum(ensemble) / TEST_SIZE)
    metr.append(ece)
    metr.append(waic)
    metr.append(var)
    print(np.sum(ensemble) / TEST_SIZE, 'ensemble')
    return metr



print("Classes loaded")

nll_several_runs = []
loss_several_runs = []
metrics_several_runs = []


#now do K-fold CV


X = full_data[:,0:p]
y = full_data[:,-1]
skf = StratifiedKFold(n_splits=10,shuffle = True,random_state = 1)
for i, (train_index, test_index) in enumerate(skf.split(X, y)):
    print('network', i)
    torch.manual_seed(i)
    net = Net().to(DEVICE)
    optimizer = optim.Adam(net.parameters(),lr = lr)
    all_nll = []
    all_loss = []
   
    X_train,y_train = X[train_index],y[train_index]
    X_test,y_test = X[test_index],y[test_index]
    
    mu,std = X_train.mean(axis = 0), X_train.std(axis = 0)
    X_tr = (X_train - mu) / std
    X_te = (X_test  - mu) / std
    
    train_dat = torch.tensor(np.column_stack((X_tr,y_train)),dtype = torch.float32)
    test_dat = torch.tensor(np.column_stack((X_te,y_test)),dtype = torch.float32)

    mu,std = X_train.mean(axis = 0), X_train.std(axis = 0)
    
    for epoch in range(epochs):
        print('epoch', epoch)
        nll, loss = train(net,train_dat, optimizer,BATCH_SIZE)

        all_nll.append(nll)
        all_loss.append(loss)
        
    nll_several_runs.append(all_nll)
    loss_several_runs.append(all_loss)
    metrics = test_ensemble(net,test_dat)
    metrics_several_runs.append(metrics)
      
current_dir = os.getcwd()
savepath = current_dir +'/results/dry_beans_BNN_metrics.txt'
    
np.savetxt(savepath, metrics_several_runs, delimiter=',')    
m = np.array(metrics_several_runs)  
np.savetxt(current_dir +'/results/annMCDROP_avg_metrics.txt',m.mean(axis = 0),delimiter = ',')
print(m.min(axis =0),m.mean(axis = 0),m.max(axis = 0))
    

    
    


    
