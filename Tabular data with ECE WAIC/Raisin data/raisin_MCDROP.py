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
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "mps")
LOADER_KWARGS = {'num_workers': 1, 'pin_memory': True} if torch.cuda.is_available() else {}



BATCH_SIZE = 810
TEST_BATCH_SIZE = 90 # number of test samples
CLASSES = 2

TRAIN_SIZE = 810
TEST_SIZE = 90
NUM_BATCHES = TRAIN_SIZE/BATCH_SIZE


TEST_SAMPLES = config['test_samples']
epochs = config['num_epochs']
dim = config['hidden_dim']
lr = config['lr']



NUM_BATCHES = TRAIN_SIZE/BATCH_SIZE




assert (TRAIN_SIZE % BATCH_SIZE) == 0
assert (TEST_SIZE % TEST_BATCH_SIZE) == 0

import pandas as pd
df = pd.read_excel('Raisin_Dataset.xlsx')
y = df['Class']
y = y.replace('Kecimen',1)
y = y.replace('Besni',0)
df.drop('Class', inplace=True, axis=1)
full_data = np.column_stack((df,y))


p = full_data.shape[1] - 1

class Net(nn.Module):
  def __init__(self):
    super(Net, self).__init__()
    
    self.fc1 = nn.Linear(p, dim)
    self.fc2 = nn.Linear(dim, 1)
    self.Dropout = nn.Dropout(0.9) #corresponding to 0.1 inclusion rate for LBBNNs
    self.loss = nn.BCELoss(reduction='sum')
    self.act = nn.ReLU()
  
  def forward(self, x, sample=False):
      x = self.act(self.fc1(x))
      x = self.Dropout(x)
      x = torch.sigmoid(self.fc2(x))

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
        target = _y.to(DEVICE)
        target = target.unsqueeze(1).float()
        net.zero_grad()
        outputs = net(data)
        negative_log_likelihood = net.loss(outputs, target)
        loss = negative_log_likelihood
        loss.backward()
        optimizer.step()
    print('loss', loss.item())
    print('nll', negative_log_likelihood.item())
    return negative_log_likelihood.item(), loss.item()



def ece_score(output, label, n_bins=10):

    pypy = np.array(output)
    y_test = np.array(label,dtype = int)
   
    py = np.zeros((pypy.shape[0],2))
    for i, prob in enumerate(pypy.squeeze()): #need two dimension for predicted probs
        py[i,1] = prob
        py[i,0] = 1 - prob
          
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



def enable_dropout(model):
    """ Function to enable the dropout layers during test-time """
    for m in model.modules():
        if m.__class__.__name__.startswith('Dropout'):
            m.train()
            
            
def test_ensemble(net,test_data):
    net.eval()
    enable_dropout(net)
    metr = []
    ensemble = []
    ece = []
    waic = []
    nll = nn.BCELoss(reduction='none')
    with torch.no_grad():
        old_batch = 0
        for batch in range(int(np.ceil(test_data.shape[0] / TEST_BATCH_SIZE))):
            batch = (batch + 1)
            _x = test_data[old_batch: TEST_BATCH_SIZE * batch, 0:p]
            _y = test_data[old_batch: TEST_BATCH_SIZE * batch, -1]

            old_batch = TEST_BATCH_SIZE * batch

            data = _x.to(DEVICE)
            target = _y.to(DEVICE)
            outputs = torch.zeros(TEST_SAMPLES, TEST_BATCH_SIZE, 1).to(DEVICE)
            logliks = torch.zeros(TEST_SAMPLES,TEST_BATCH_SIZE)
            for i in range(TEST_SAMPLES):
               # enable_dropout(net)
                outputs[i] = net(data)
                t = target.unsqueeze(1)
                logliks[i] = - nll(outputs[i],t).squeeze()
                
    
            var = logliks.var(axis = 0).sum()
    
        

            output1 = outputs.mean(0)

            class_pred = output1.round().squeeze()
            
            tar = target.detach().cpu().numpy()
    
           
            o= output1.detach().cpu().numpy()
       
        
            ece  = ece_score(o,tar)
            
            
            ##for full model averaging
            likelihoods = torch.exp(logliks)
            first_term = torch.log(likelihoods.mean(axis = 0))
            second_term = logliks.mean(axis = 0)
            
            waic = 2 * torch.sum(first_term - second_term)
            
         
        
            a = ((class_pred == target) * 1).sum().item()
         
    
            ensemble.append(a)
    metr.append(np.sum(ensemble) / TEST_SIZE)
    metr.append(ece)
    metr.append(waic)
    metr.append(var)
    print(np.sum(ensemble) / TEST_SIZE, 'ensemble')
    return metr


nll_several_runs = []
loss_several_runs = []
metrics_several_runs = []


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
savepath = current_dir +'/results/raisin_ANNMCDROP_metrics.txt'
    
np.savetxt(savepath, metrics_several_runs, delimiter=',')    

m = np.array(metrics_several_runs)
np.savetxt(current_dir +'/results/annMCDROP_avg_metrics.txt',m.mean(axis = 0),delimiter = ',')
print(m.min(axis = 0),m.mean(axis = 0),m.max(axis = 0))