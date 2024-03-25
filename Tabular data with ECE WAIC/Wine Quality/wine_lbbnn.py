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
from lbbnn_layers import BayesianLinear


# select the device
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
LOADER_KWARGS = {'num_workers': 1, 'pin_memory': True} if torch.cuda.is_available() else {}

from ucimlrepo import fetch_ucirepo 

wine_quality = fetch_ucirepo(id=186) 
  
# data (as pandas dataframes) 
dat = wine_quality.data.features 
target = wine_quality.data.targets 


X = dat.drop([6490,6491,6492,6493,6494,6495,6496])
y = target.drop([6490,6491,6492,6493,6494,6495,6496])


TRAIN_SIZE = 5841
TEST_SIZE =  649
BATCH_SIZE = 5841
TEST_BATCH_SIZE = 649

NUM_BATCHES = TRAIN_SIZE/BATCH_SIZE

TEST_SAMPLES = config['test_samples']
epochs = config['num_epochs']
dim = config['hidden_dim']
lr = config['lr']
SAMPLES = 1

assert (TRAIN_SIZE % BATCH_SIZE) == 0
assert (TEST_SIZE % TEST_BATCH_SIZE) == 0

class BayesianNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        # set the architecture
        self.l1 = BayesianLinear(X.shape[1], dim)
        self.l2 = BayesianLinear(dim, 1)
        self.loss  = nn.GaussianNLLLoss(reduction='sum')
        self.act = nn.ReLU()
        
    def forward(self, x,g1,g2, sample=False):
        x = x.view(-1, X.shape[1])
        x = self.act(self.l1(x,g1, sample))
        x = self.l2(x,g2, sample)

        return x

    def log_prior(self):
        return self.l1.log_prior \
               + self.l2.log_prior 

    def log_variational_posterior(self):
        return self.l1.log_variational_posterior \
               + self.l2.log_variational_posterior 

    # sample the marginal likelihood lower bound
    def sample_elbo(self, input, target, samples=SAMPLES):
        outputs = torch.zeros(samples, BATCH_SIZE, 1).to(DEVICE)
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
            var = torch.ones(size = target.shape).to(DEVICE)
            negative_log_likelihoods[i] = self.loss(outputs[i], target,var)

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
        _x = train_data[old_batch: batch_size * batch,0:X.shape[1]]
        _y = train_data[old_batch: batch_size * batch, -1]
        
        old_batch = batch_size * batch
        data = _x.to(DEVICE)
        target = _y.to(DEVICE)
        target = target.unsqueeze(1).float()
        net.zero_grad()
        loss, log_prior, log_variational_posterior, negative_log_likelihood = net.sample_elbo(data, target)
        loss.backward()
        optimizer.step()
    print('loss', loss.item())
    print('nll', negative_log_likelihood.item())
    return negative_log_likelihood.item(), loss.item()


def pinball_loss(y_true,y_pred):
    alpha = np.arange(0.05,1.00,0.05) #from 0.05 -> 0.95 in 0.05 increments
    loss = np.zeros(len(alpha))
    for i,a in enumerate(alpha):
        loss[i] = mean_pinball_loss(y_true, y_pred,alpha = a)
        
    
    return loss.mean()



def test_ensemble(net,test_data):
    net.eval()
    metr = []
    density = np.zeros(TEST_SAMPLES)

    rmse = []
    rmse_mpm = []
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
            outputs_mpm = torch.zeros_like(outputs)
            logliks = torch.zeros(TEST_SAMPLES,TEST_BATCH_SIZE).to(DEVICE)
            logliks_mpm = torch.zeros_like(logliks)
           
            for i in range(TEST_SAMPLES):
                net.l1.alpha = 1 / (1 + torch.exp(-net.l1.lambdal))
                net.l1.gamma.alpha = net.l1.alpha
                net.l2.alpha = 1 / (1 + torch.exp(-net.l2.lambdal))
                net.l2.gamma.alpha = net.l2.alpha
            
                # sample the model
                cgamma1 = net.l1.gamma.rsample().to(DEVICE)
                cgamma2 = net.l2.gamma.rsample().to(DEVICE)
           
                cg3 = (net.l1.alpha > 0.5) * 1
                cg4 = (net.l2.alpha > 0.5) * 1
                outputs[i] = net.forward(data, g1=cgamma1, g2=cgamma2, sample=True)
                outputs_mpm[i] = net.forward(data, g1=cg3, g2=cg4, sample=False)
                t = target.unsqueeze(1)
                logliks[i] = - nll(outputs[i],t,torch.ones(size = t.shape).to(DEVICE)).squeeze()
                logliks_mpm[i] = - nll(outputs_mpm[i],t,torch.ones(size = t.shape).to(DEVICE)).squeeze()
                
                
    
                ## sample the inclusion variables for each layer to estimate the sparsity level
                g1 = np.random.binomial(n=1, p=net.l1.alpha.detach().cpu().numpy())
                g2 = np.random.binomial(n=1, p=net.l2.alpha.detach().cpu().numpy())
            

                gammas = np.concatenate((g1.flatten(), g2.flatten()))
                density[i] = gammas.mean() #compute density for each model in the ensemble

            output1 = outputs.mean(0)
            outputs_mpm = outputs_mpm.mean(0)
            
            pinball = pinball_loss(target.detach().cpu().numpy(),output1.detach().cpu().numpy())
            pinball_mpm = pinball_loss(target.detach().cpu().numpy(),outputs_mpm.detach().cpu().numpy())
            
            RMSE = torch.sqrt(crit(output1.squeeze(),target))
            rmse2 = torch.sqrt(crit(outputs_mpm.squeeze(),target))
            rmse.append(RMSE.detach().cpu().numpy())
            rmse_mpm.append(rmse2.detach().cpu().numpy())
            
            
            var = logliks.var(axis = 0).sum()
            var_mpm = logliks_mpm.var(axis = 0).sum()

        
            
           
   
            likelihoods = torch.exp(logliks)
            first_term = torch.log(likelihoods.mean(axis = 0))
            second_term = logliks.mean(axis = 0)
            waic = 2 * torch.sum(first_term - second_term)
            
            likelihoods2 = torch.exp(logliks_mpm)
            first_term2 = torch.log(likelihoods2.mean(axis = 0))
            second_term2 = logliks_mpm.mean(axis = 0)
            waic_mpm = 2 * torch.sum(first_term2 - second_term2)

           
            
          


    metr.append(np.mean(rmse))
    metr.append(np.mean(rmse_mpm))
    metr.append(np.mean(density))
    metr.append(pinball)
    metr.append(pinball_mpm)
    metr.append(waic.detach().cpu().numpy())
    metr.append(waic_mpm.detach().cpu().numpy())
    metr.append(var.detach().cpu().numpy())
    metr.append(var_mpm.detach().cpu().numpy())
    print(np.mean(rmse), 'rmse')
    print(np.mean(rmse_mpm), 'rmse_mpm')
    return metr



print("Classes loaded")

nll_several_runs = []
loss_several_runs = []
metrics_several_runs = []


#now do K-fold CV


X = np.array(X)
y = np.array(y)

skf =KFold(n_splits=10,shuffle = True,random_state = 1)
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
                {'params': net.l2.lambdal, 'lr': 0.0005}], lr=0.0005)
        nll, loss = train(net,train_dat, optimizer,BATCH_SIZE)
        all_nll.append(nll)
        all_loss.append(loss)
        
    nll_several_runs.append(all_nll)
    loss_several_runs.append(all_loss)
    metrics = test_ensemble(net,test_dat)
    metrics_several_runs.append(metrics)
      
current_dir = os.getcwd()
savepath = current_dir +'/results/wine_lbbnn_metrics.txt'
    
np.savetxt(savepath, metrics_several_runs, delimiter=',')    
m = np.array(metrics_several_runs)
np.savetxt(current_dir +'/results/lbbnn_lbbnn_metrics.txt',m.mean(axis = 0),delimiter = ',')
    

print(m.min(axis = 0),m.mean(axis = 0),m.max(axis = 0))

print(m[:,0].min(),m[:,0].mean(),m[:,0].max(),'rmse')
print(m[:,1].min(),m[:,1].mean(),m[:,1].max(),'rmse_mpm')
print(m[:,2].min(),m[:,2].mean(),m[:,2].max(),'density')
print(m[:,3].min(),m[:,3].mean(),m[:,3].max(),'pinball')
print(m[:,4].min(),m[:,4].mean(),m[:,4].max(),'pinball_mpm')
print(m[:,5].min(),m[:,5].mean(),m[:,5].max(),'waic')
print(m[:,6].min(),m[:,6].mean(),m[:,6].max(),'waic_mpm')
print(m[:,7].min(),m[:,7].mean(),m[:,7].max(),'var')
print(m[:,8].min(),m[:,8].mean(),m[:,8].max(),'var_mpm')








