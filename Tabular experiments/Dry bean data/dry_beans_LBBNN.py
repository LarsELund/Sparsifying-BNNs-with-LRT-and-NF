#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import os
from sklearn.model_selection import StratifiedKFold




# select the device
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
LOADER_KWARGS = {'num_workers': 1, 'pin_memory': True} if torch.cuda.is_available() else {}

if (torch.cuda.is_available()):
    print("GPUs are used!")
else:
    print("CPUs are used!")

BATCH_SIZE = 12240
TEST_BATCH_SIZE = 1360 # number of test samples
CLASSES = 7

TRAIN_SIZE = 12240
TEST_SIZE = 1360
NUM_BATCHES = TRAIN_SIZE/BATCH_SIZE
SAMPLES = 1
TEMPER_PRIOR = 0.001

#labs = (Seker, Barbunya, Bombay, Cali, Dermosan, Horoz, Sira) 

import sys



current_dir = os.getcwd()
sys.path.append('../layers')
sys.path.append('../config')
from config import config
from lbbnn_layers import BayesianLinear

TEST_SAMPLES = config['test_samples']
epochs = config['num_epochs']
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




class BayesianNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        # set the architecture
        self.l1 = BayesianLinear(p, dim)
        self.l2 = BayesianLinear(dim, 7)

        
        
    def forward(self, x,g1,g2, sample=False):
        x = x.view(-1, p)
        x = F.relu(self.l1(x,g1, sample))
        x = F.log_softmax((self.l2.forward(x,g2 ,sample)), dim=1)
        

        return x

    def log_prior(self):
        return self.l1.log_prior \
               + self.l2.log_prior 

    def log_variational_posterior(self):
        return self.l1.log_variational_posterior \
               + self.l2.log_variational_posterior 

    # sample the marginal likelihood lower bound
    def sample_elbo(self, input, target, samples=SAMPLES):
        outputs = torch.zeros(samples, BATCH_SIZE, CLASSES).to(DEVICE)
        log_priors = torch.zeros(samples).to(DEVICE)
        log_variational_posteriors = torch.zeros(samples).to(DEVICE)
        negative_log_likelihoods = torch.zeros(samples).to(DEVICE)
        for i in range(samples):
            # get the inclusion probabilities for all layers
            self.l1.alpha = 1 / (1 + torch.exp(-self.l1.lambdal))
            self.l1.gamma.alpha = self.l1.alpha
            self.l2.alpha = 1 / (1 + torch.exp(-self.l2.lambdal))
            self.l2.gamma.alpha = self.l2.alpha
         

            # sample the model
            cgamma1 = self.l1.gamma.rsample().to(DEVICE)
            cgamma2 = self.l2.gamma.rsample().to(DEVICE)
        

            # get the results
            outputs[i] = self.forward(input, g1=cgamma1, g2=cgamma2, sample=True)
            log_priors[i] = self.log_prior()
            log_variational_posteriors[i] = self.log_variational_posterior()
            negative_log_likelihoods[i] = F.nll_loss(outputs[i], target, reduction='sum')

        # the current log prior
        log_prior = log_priors.mean()
        # the current log variational posterior
        log_variational_posterior = log_variational_posteriors.mean()
        # the current negative log likelihood
        negative_log_likelihood = negative_log_likelihoods.mean()

        # the current ELBO
        loss = negative_log_likelihood + (log_variational_posterior - log_prior) / NUM_BATCHES
        return loss, log_prior, log_variational_posterior, negative_log_likelihood

def train(net,train_data, optimizer, batch_size = BATCH_SIZE):
    net.train()
    old_batch = 0
    for batch in range(int(np.ceil(train_data.shape[0] / batch_size))):
        batch = (batch + 1)
        _x = train_data[old_batch: batch_size * batch,0:p]
        _y = train_data[old_batch: batch_size * batch, -1]
        
        old_batch = batch_size * batch
        data = _x.to(DEVICE)
        target = _y.to(torch.long)
        net.zero_grad()
        loss, log_prior, log_variational_posterior, negative_log_likelihood = net.sample_elbo(data, target)
        loss.backward()
        optimizer.step()
    print('loss', loss.item())
    print('nll', negative_log_likelihood.item())
    return negative_log_likelihood.item(), loss.item()

def test_ensemble(net,test_data):
    net.eval()
    metr = []
    density = np.zeros(TEST_SAMPLES)
    ensemble = []
    with torch.no_grad():
        old_batch = 0
        for batch in range(int(np.ceil(test_data.shape[0] / TEST_BATCH_SIZE))):
            batch = (batch + 1)
            _x = test_data[old_batch: TEST_BATCH_SIZE * batch, 0:p]
            _y = test_data[old_batch: TEST_BATCH_SIZE * batch, -1]

            old_batch = TEST_BATCH_SIZE * batch

            data = _x.to(DEVICE)
            target = _y.to(torch.long)
            outputs = torch.zeros(TEST_SAMPLES, TEST_BATCH_SIZE, CLASSES).to(DEVICE)
            for i in range(TEST_SAMPLES):
                net.l1.alpha = 1 / (1 + torch.exp(-net.l1.lambdal))
                net.l1.gamma.alpha = net.l1.alpha
                net.l2.alpha = 1 / (1 + torch.exp(-net.l2.lambdal))
                net.l2.gamma.alpha = net.l2.alpha
           
                # sample the model
                cgamma1 = net.l1.gamma.rsample().to(DEVICE)
                cgamma2 = net.l2.gamma.rsample().to(DEVICE)
           
                
                outputs[i] = net.forward(data, g1=cgamma1, g2=cgamma2, sample=True)
    
                ## sample the inclusion variables for each layer to estimate the sparsity level
                g1 = np.random.binomial(n=1, p=net.l1.alpha.detach().cpu().numpy())
                g2 = np.random.binomial(n=1, p=net.l2.alpha.detach().cpu().numpy())
           

                gammas = np.concatenate((g1.flatten(), g2.flatten()))
                density[i] = gammas.mean() #compute density for each model in the ensemble

            output1 = outputs.mean(0)
            pred1 = output1.max(1, keepdim=True)[1]  # index of max log-probability
        


         
            b = pred1.eq(target.view_as(pred1)).sum().item()
            ensemble.append(b)



    metr.append(np.sum(ensemble) / TEST_SIZE)
    print(np.mean(density), 'density')
    metr.append(np.mean(density))
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
    optimizer = optim.Adam([
        {'params': net.l1.bias_mu, 'lr': 0.005},
        {'params': net.l2.bias_mu, 'lr': 0.005},
      
        {'params': net.l1.bias_rho, 'lr': 0.005},
        {'params': net.l2.bias_rho, 'lr': 0.005},
     
        {'params': net.l1.weight_mu, 'lr': 0.005},
        {'params': net.l2.weight_mu, 'lr': 0.005},
     
        {'params': net.l1.weight_rho, 'lr': 0.005},
        {'params': net.l2.weight_rho, 'lr': 0.005},
  
        {'params': net.l1.pa, 'lr': 0.1},
        {'params': net.l2.pa, 'lr': 0.1},
 
        {'params': net.l1.pb, 'lr': 0.1},
        {'params': net.l2.pb, 'lr': 0.1},
 
        {'params': net.l1.weight_a, 'lr': 0.005},
        {'params': net.l2.weight_a, 'lr': 0.005},
 
        {'params': net.l1.weight_b, 'lr': 0.005},
        {'params': net.l2.weight_b, 'lr': 0.005},
    
        {'params': net.l1.bias_a, 'lr': 0.005},
        {'params': net.l2.bias_a, 'lr': 0.005},
    
        {'params': net.l1.bias_b, 'lr': 0.005},
        {'params': net.l2.bias_b, 'lr': 0.005},

        {'params': net.l1.lambdal, 'lr': 0.1},
        {'params': net.l2.lambdal, 'lr': 0.1}
   
    ], lr=0.005)
    
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
        if (net.l1.pa / (net.l1.pa + net.l1.pb)).mean() < 0.1 or epoch == 200:
            print(epoch)
            net.l1.gamma_prior.exact = True
            net.l2.gamma_prior.exact = True
         
            net.l1.bias_prior.exact = True
            net.l2.bias_prior.exact = True
          
            net.l1.weight_prior.exact = True
            net.l2.weight_prior.exact = True
       
            optimizer = optim.Adam([
                {'params': net.l1.bias_mu, 'lr': 0.0005},
                {'params': net.l2.bias_mu, 'lr': 0.0005},
               
                {'params': net.l1.bias_rho, 'lr': 0.0005},
                {'params': net.l2.bias_rho, 'lr': 0.0005},
             
                {'params': net.l1.weight_mu, 'lr': 0.0005},
                {'params': net.l2.weight_mu, 'lr': 0.0005},
             
                {'params': net.l1.weight_rho, 'lr': 0.0005},
                {'params': net.l2.weight_rho, 'lr': 0.0005},
             
                {'params': net.l1.pa, 'lr': 0.00},
                {'params': net.l2.pa, 'lr': 0.00},
             
                {'params': net.l1.pb, 'lr': 0.00},
                {'params': net.l2.pb, 'lr': 0.00},
       
                {'params': net.l1.weight_a, 'lr': 0.00},
                {'params': net.l2.weight_a, 'lr': 0.00},
              
                {'params': net.l1.weight_b, 'lr': 0.00},
                {'params': net.l2.weight_b, 'lr': 0.00},
          
                {'params': net.l1.bias_a, 'lr': 0.00},
                {'params': net.l2.bias_a, 'lr': 0.00},
       
                {'params': net.l1.bias_b, 'lr': 0.00},
                {'params': net.l2.bias_b, 'lr': 0.00},
    
                {'params': net.l1.lambdal, 'lr': 0.0005},
                {'params': net.l2.lambdal, 'lr': 0.0005}
               
            ], lr=0.0005)
        nll, loss = train(net,train_dat, optimizer,BATCH_SIZE)
        all_nll.append(nll)
        all_loss.append(loss)
        
    nll_several_runs.append(all_nll)
    loss_several_runs.append(all_loss)
    metrics = test_ensemble(net,test_dat)
    metrics_several_runs.append(metrics)
      
current_dir = os.getcwd()
savepath = current_dir +'/results/dry_beans_LBBNN_metrics.txt'
    
np.savetxt(savepath, metrics_several_runs, delimiter=',')    

m = np.array(metrics_several_runs)  
np.savetxt(current_dir +'/results/lbbnn_avg_metrics.txt',m.mean(axis = 0),delimiter = ',')
    
print(m.min(axis =0),m.mean(axis = 0),m.max(axis = 0))

