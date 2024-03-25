#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import math
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import os
from sklearn.model_selection import KFold
import pandas as pd
from torch.optim.lr_scheduler import MultiStepLR
from sklearn.metrics import mean_pinball_loss
from sklearn import preprocessing


import sys
current_dir = os.getcwd()
sys.path.append('../layers')
sys.path.append('../config')
from config import config
from mnf_layers import BayesianLinear


# select the device
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "mps")
LOADER_KWARGS = {'num_workers': 1, 'pin_memory': True} if torch.cuda.is_available() else {}

from ucimlrepo import fetch_ucirepo 

  
# fetch dataset 
abalone = fetch_ucirepo(id=1) 
  
# data (as pandas dataframes) 
dat = abalone.data.features 
target = abalone.data.targets 

X = dat.drop([4170,4171,4172,4173,4174,4175,4176])
y = target.drop([4170,4171,4172,4173,4174,4175,4176])

for label in "MFI":
   X[label] = X["Sex"] == label 
   X[label] *= 1
del X["Sex"]





TRAIN_SIZE = 3753
TEST_SIZE = 417
BATCH_SIZE = 3753
TEST_BATCH_SIZE = 417

NUM_BATCHES = TRAIN_SIZE/BATCH_SIZE

TEST_SAMPLES = config['test_samples']
epochs = config['num_epochs']
dim = config['hidden_dim']
lr = config['lr']
num_transforms = config['num_transforms']







assert (TRAIN_SIZE % BATCH_SIZE) == 0
assert (TEST_SIZE % TEST_BATCH_SIZE) == 0


# deine the whole BNN
class BayesianNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        # set the architecture
        self.l1 = BayesianLinear(X.shape[1], dim,num_transforms = num_transforms)
        self.l2 = BayesianLinear(dim, 1,num_transforms = num_transforms)
        
        
    def forward(self, x, sample=False):
        x = x.view(-1, X.shape[1])
        x = F.relu(self.l1(x, sample))
        x = self.l2(x, sample)

        return x

    def kl(self):
        return self.l1.kl_div() + self.l2.kl_div() 
    
    
    
# Stochastic Variational Inference iteration
def train(net,train_data, optimizer, batch_size = BATCH_SIZE):
    net.train()
    old_batch = 0
    for batch in range(int(np.ceil(train_data.shape[0] / batch_size))):
        batch = (batch + 1)
        _x = train_data[old_batch: batch_size * batch,0:X.shape[1]]
        _y = train_data[old_batch: batch_size * batch, -1]
        
        old_batch = batch_size * batch
        data = _x.to(DEVICE)
        target = _y.to(DEVICE)
        target = target.unsqueeze(1).float()
        net.zero_grad()
        outputs = net(data)
        criterion = nn.GaussianNLLLoss(reduction='sum')
        var = torch.ones(size = outputs.shape).to(DEVICE)
        loss = criterion(outputs, target,var) + net.kl() / NUM_BATCHES
        loss.backward()
        optimizer.step()
    print('loss', loss.item())

    return loss.item()

def pinball_loss(y_true,y_pred):
    alpha = np.arange(0.05,1.00,0.05) #from 0.05 -> 0.95 in 0.05 increments
    loss = np.zeros(len(alpha))
    for i,a in enumerate(alpha):
        loss[i] = mean_pinball_loss(y_true, y_pred,alpha = a)
        
    
    return loss.mean()






def test_ensemble(net,test_data):
    net.eval()
    metr = []
    ensemble = []
    waic = []
    rmse = []
  
    crit = nn.MSELoss(reduction='mean')
    nll = nn.GaussianNLLLoss(reduction='none')
    with torch.no_grad():
        old_batch = 0
        for batch in range(int(np.ceil(test_data.shape[0] / TEST_BATCH_SIZE))):
            batch = (batch + 1)
            _x = test_data[old_batch: TEST_BATCH_SIZE * batch, 0:X.shape[1]]
            _y = test_data[old_batch: TEST_BATCH_SIZE * batch, -1]

            old_batch = TEST_BATCH_SIZE * batch

            data = _x.to(DEVICE)
            target = _y.to(DEVICE)
            outputs = torch.zeros(TEST_SAMPLES, TEST_BATCH_SIZE, 1).to(DEVICE)
            logliks = torch.zeros(TEST_SAMPLES,TEST_BATCH_SIZE).to(DEVICE)
            for i in range(TEST_SAMPLES):
                outputs[i] = net(data, sample=True)
                t = target.unsqueeze(1)
                logliks[i] = - nll(outputs[i],t,torch.ones(size = t.shape).to(DEVICE)).squeeze()
             
    

            output1 = outputs.mean(0)
            pinball = pinball_loss(target.detach().cpu().numpy(),output1.detach().cpu().numpy())
            
            RMSE = torch.sqrt(crit(output1.squeeze(),target))
            rmse.append(RMSE.detach().cpu().numpy())
            
            var = logliks.var(axis = 0).sum()

        
            
           
   
            likelihoods = torch.exp(logliks)
            first_term = torch.log(likelihoods.mean(axis = 0))
            second_term = logliks.mean(axis = 0)
            
            waic = 2 * torch.sum(first_term - second_term)
        
           
    metr.append(np.mean(rmse))
    metr.append(pinball)
    metr.append(waic.detach().cpu().numpy())
    metr.append(var.detach().cpu().numpy())
    print(np.mean(rmse), 'rmse')
    return metr

print("Classes loaded")

nll_several_runs = []
loss_several_runs = []
metrics_several_runs = []


#now do K-fold CV


X = np.array(X)
y = np.array(y)

skf = KFold(n_splits=10,shuffle = True,random_state = 1)
for i, (train_index, test_index) in enumerate(skf.split(X, y)):
    print('network', i)
    torch.manual_seed(i)
    net = BayesianNetwork().to(DEVICE)
    optimizer = optim.Adam(net.parameters(),lr = lr)
    
    all_nll = []
    all_loss = []
   
    X_train,y_train = X[train_index],y[train_index]
    X_test,y_test = X[test_index],y[test_index]
    
    X_train,y_train = X[train_index],y[train_index]
    X_test,y_test = X[test_index],y[test_index]
    scaler = preprocessing.StandardScaler().fit(X_train)
    y_scaler = preprocessing.StandardScaler().fit(y_train)
    
   # mu,std = X_train.mean(axis = 0), X_train.std(axis = 0) #standardize all columns
    X_tr = scaler.transform(X_train)
    X_te = scaler.transform(X_test)
    y_tr = y_scaler.transform(y_train)
    y_te = y_scaler.transform(y_test)
    
    
    train_dat = torch.tensor(np.column_stack((X_tr,y_tr)),dtype = torch.float32)
    test_dat = torch.tensor(np.column_stack((X_te,y_te)),dtype = torch.float32)
    
    
    
    for epoch in range(epochs):
        print('epoch', epoch)
        loss = train(net,train_dat, optimizer,BATCH_SIZE)
       
        all_loss.append(loss)
        
    nll_several_runs.append(all_nll)
    loss_several_runs.append(all_loss)
    metrics = test_ensemble(net,test_dat)
    metrics_several_runs.append(metrics)
      
current_dir = os.getcwd()
savepath = current_dir +'/results/abalone_MNF_metrics.txt'
    
np.savetxt(savepath, metrics_several_runs, delimiter=',')    
m = np.array(metrics_several_runs)
np.savetxt(current_dir +'/results/mnf_avg_metrics.txt',m.mean(axis = 0),delimiter = ',')
    

print(m.min(axis = 0),m.mean(axis = 0),m.max(axis = 0))

print(m[:,0].min(),m[:,0].mean(),m[:,0].max(),'rmse')
print(m[:,1].min(),m[:,1].mean(),m[:,1].max(),'pinball')
print(m[:,2].min(),m[:,2].mean(),m[:,2].max(),'waic')
print(m[:,3].min(),m[:,3].mean(),m[:,3].max(),'var')




