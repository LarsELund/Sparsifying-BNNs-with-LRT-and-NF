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



# define the parameters
BATCH_SIZE = 621
TEST_BATCH_SIZE = 69
CLASSES = 2

TEST_SAMPLES = config['test_samples']
epochs = config['num_epochs']
num_transforms = config['num_transforms']
dim = config['hidden_dim']
lr = config['lr']

full_data = np.loadtxt('dat.txt',dtype = float,delimiter = ',')
p = full_data.shape[1] - 1





TRAIN_SIZE = 621
TEST_SIZE = 69
NUM_BATCHES = TRAIN_SIZE/BATCH_SIZE




assert (TRAIN_SIZE % BATCH_SIZE) == 0
assert (TEST_SIZE % TEST_BATCH_SIZE) == 0


# define the linear layer for the BNN




# deine the whole BNN
class BayesianNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        # set the architecture
        self.l1 = nn.Linear(p, dim)
        self.l2 = nn.Linear(dim, 1)
        self.loss = nn.BCELoss(reduction='sum')
        self.act = nn.ReLU()
        
    def forward(self, x):
        x = x.view(-1, p)
        x = F.relu(self.l1(x))
        x = torch.sigmoid(self.l2(x))

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


def test_ensemble(net,test_data):
    net.eval()
    metr = []
    ensemble = []
    with torch.no_grad():
        old_batch = 0
        for batch in range(int(np.ceil(test_data.shape[0] / TEST_BATCH_SIZE))):
            batch = (batch + 1)
            _x = test_data[old_batch: TEST_BATCH_SIZE * batch, 0:p]
            _y = test_data[old_batch: TEST_BATCH_SIZE * batch, -1]

            old_batch = TEST_BATCH_SIZE * batch

            data = _x.to(DEVICE)
            target = _y.to(DEVICE)

        

            output1 = net(data)

            class_pred = output1.round().squeeze()
         
        
            a = ((class_pred == target) * 1).sum().item()
         
    
            ensemble.append(a)
    metr.append(np.sum(ensemble) / TEST_SIZE)
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
    net = BayesianNetwork().to(DEVICE)
    optimizer = optim.Adam(net.parameters(),lr = lr,weight_decay=0.0)
    
    all_nll = []
    all_loss = []
   
    X_train,y_train = X[train_index],y[train_index]
    X_test,y_test = X[test_index],y[test_index]
    train_dat = torch.tensor(np.column_stack((X_train,y_train)),dtype = torch.float32)
    test_dat = torch.tensor(np.column_stack((X_test,y_test)),dtype = torch.float32)
   
    m1,s1 = train_dat[:,1].mean(),train_dat[:,1].std()
    m2,s2 = train_dat[:,2].mean(),train_dat[:,2].std()
    m7,s7 = train_dat[:,7].mean(),train_dat[:,7].std()
    m13,s13 = train_dat[:,13].mean(),train_dat[:,13].std()
    m14,s14 = train_dat[:,14].mean(),train_dat[:,14].std()


    train_dat[:,1] = (train_dat[:,1] - m1) / s1
    train_dat[:,2] = (train_dat[:,2] - m2) / s2
    train_dat[:,7] = (train_dat[:,7] - m7) / s7
    train_dat[:,13] = (train_dat[:,13] - m13) / s13
    train_dat[:,14] = (train_dat[:,14] - m14) / s14

    test_dat[:,1] = (test_dat[:,1] - m1) / s1
    test_dat[:,2] = (test_dat[:,2] - m2) / s2
    test_dat[:,7] = (test_dat[:,7] - m7) / s7
    test_dat[:,13] = (test_dat[:,13] - m13) / s13
    test_dat[:,14] = (test_dat[:,14] - m14) / s14
    
    for epoch in range(epochs):
        print('epoch', epoch)
        nll, loss = train(net,train_dat, optimizer,BATCH_SIZE)
        optimizer.step()
        all_nll.append(nll)
        all_loss.append(loss)
        
    nll_several_runs.append(all_nll)
    loss_several_runs.append(all_loss)
    metrics = test_ensemble(net,test_dat)
    metrics_several_runs.append(metrics)
      
current_dir = os.getcwd()
savepath = current_dir +'/results/credit_approval_ANN_metrics.txt'
    
np.savetxt(savepath, metrics_several_runs, delimiter=',')    

m = np.array(metrics_several_runs)
np.savetxt(current_dir +'/results/ann_avg_metrics.txt',np.mean(m,axis = 0),delimiter = ',')
print(m.min(axis = 0),m.mean(axis = 0),m.max(axis = 0))