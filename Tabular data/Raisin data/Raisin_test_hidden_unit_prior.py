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
epochs = 500
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

# define the linear layer for the BNN



class BayesianLinear(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        # weight variational parameters
        self.weight_mu = nn.Parameter(torch.Tensor(out_features, in_features).uniform_(-0.2, 0.2))
        self.weight_rho = nn.Parameter(-9 + 1. * torch.randn(out_features,in_features))
        self.weight_sigma = torch.empty(size=self.weight_rho.shape)

        # weight priors = N(0,1)
        self.mu_prior = torch.zeros(out_features, device=DEVICE)
        self.sigma_prior = (self.mu_prior + 1).to(DEVICE)
            

        # bias variational parameters
        self.bias_mu = nn.Parameter(torch.Tensor(out_features).uniform_(-0.2, 0.2))
        self.bias_rho = nn.Parameter(-9+ 1 * torch.randn(out_features))
        self.bias_sigma = torch.empty(self.bias_rho.shape)

      
        # scalars
        self.kl = 0
        self.z = 0
        self.shift = 0


    # forward path
    def forward(self, input, sample=False, calculate_log_probs=False):
        self.weight_sigma = torch.log1p(torch.exp(self.weight_rho))
        self.bias_sigma = torch.log1p(torch.exp(self.bias_rho))
        if self.training or sample:

            e_w = self.weight_mu 
            var_w = self.weight_sigma ** 2
            e_b = torch.mm(input, e_w.T) + self.bias_mu
            var_b = torch.mm(input ** 2, var_w.T) + self.bias_sigma ** 2
            eps = torch.randn(size=(var_b.size()), device=DEVICE)
            activations = e_b + torch.sqrt(var_b) * eps
            
           
            

            

        if self.training or calculate_log_probs:

        

        

            kl_weight =  (torch.log(self.sigma_prior / var_b.sqrt()) - 0.5 + (var_b
                    + (e_b - self.mu_prior) ** 2) / (
                               2 * self.sigma_prior ** 2)).sum()
                    

            self.kl = kl_weight 
        else:
            self.kl = 0

        return activations



# deine the whole BNN
class BayesianNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        # set the architecture
        self.l1 = BayesianLinear(p, dim)
        self.l2 = BayesianLinear(dim, 1)

        self.loss = nn.BCELoss(reduction='sum')
        self.act = nn.ReLU()
    def forward(self, x, sample=False):
        x = x.view(-1, p)
        x = self.act(self.l1(x, sample))
        x = torch.sigmoid(self.l2(x, sample))

        return x

    def kl(self):
        return self.l1.kl + self.l2.kl 

    


        


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
        outputs = net(data, sample=True)
        negative_log_likelihood = net.loss(outputs, target)
        loss = negative_log_likelihood + net.kl() / NUM_BATCHES
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
            outputs = torch.zeros(TEST_SAMPLES, TEST_BATCH_SIZE, 1).to(DEVICE)
            for i in range(TEST_SAMPLES):
                outputs[i] = net(data, sample=True)
    

            output1 = outputs.mean(0)

            class_pred = output1.round().squeeze()
         
        
            a = ((class_pred == target) * 1).sum().item()
         
    
            ensemble.append(a)
    metr.append(np.sum(ensemble) / TEST_SIZE)
    print(np.sum(ensemble) / TEST_SIZE, 'ensemble')
    return metr



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
      
#current_dir = os.getcwd()
#savepath = current_dir +'/results/raisin_BNN_metrics.txt'
    
#np.savetxt(savepath, metrics_several_runs, delimiter=',')    

m = np.array(metrics_several_runs)
#np.savetxt(current_dir +'/results/bnn_avg_metrics.txt',m.mean(axis = 0),delimiter = ',')
print(m.min(axis = 0),m.mean(axis = 0),m.max(axis = 0))