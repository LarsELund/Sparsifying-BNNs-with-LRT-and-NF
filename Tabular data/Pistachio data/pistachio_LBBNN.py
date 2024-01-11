#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import pandas as pd
df = pd.read_excel('Pistachio_28_Features_Dataset.xlsx')



import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import os
from sklearn.model_selection import StratifiedKFold
from torch.optim.lr_scheduler import MultiStepLR
import math

import sys



current_dir = os.getcwd()
sys.path.append('../layers')
sys.path.append('../config')
from config import config
from lbbnn_layers import BayesianLinear

# select the device
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
LOADER_KWARGS = {'num_workers': 1, 'pin_memory': True} if torch.cuda.is_available() else {}



# define the parameters
BATCH_SIZE = 1926
TEST_BATCH_SIZE = 214
CLASSES = 2
TRAIN_SIZE = 1926
TEST_SIZE = 214
SAMPLES = 1
TEMPER_PRIOR = 0.001
NUM_BATCHES = TRAIN_SIZE/BATCH_SIZE


TEST_SAMPLES = config['test_samples']
epochs = config['num_epochs']
dim = config['hidden_dim']
lr = config['lr']




assert (TRAIN_SIZE % BATCH_SIZE) == 0
assert (TEST_SIZE % TEST_BATCH_SIZE) == 0


import pandas as pd
df = pd.read_excel('Pistachio_28_Features_Dataset.xlsx',skiprows = 8)
y = df['Kirmizi_Pistachio']
y = y.replace('Kirmizi_Pistachio',1)
y = y.replace('Siirt_Pistachio',0)
df.drop('Kirmizi_Pistachio', inplace=True, axis=1)
full_data = np.column_stack((df,y))



p = full_data.shape[1] - 1



class BayesianNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        # set the architecture
        self.l1 = BayesianLinear(p, dim)
        self.l2 = BayesianLinear(dim, 1)
        self.loss = nn.BCELoss(reduction='sum')
        self.act = nn.LeakyReLU()
        
        
    def forward(self, x,g1,g2, sample=False):
        x = x.view(-1, p)
        x = F.relu(self.l1(x,g1, sample))
        x = torch.sigmoid(self.l2(x,g2, sample))

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
            negative_log_likelihoods[i] = self.loss(outputs[i], target)

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
        target = _y.to(DEVICE)
        target = target.unsqueeze(1).float()
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
    median = []
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
            outputs2 = torch.zeros_like(outputs)
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
                outputs2[i] = net.forward(data, g1=cg3, g2=cg4, sample=False)
    
                ## sample the inclusion variables for each layer to estimate the sparsity level
                g1 = np.random.binomial(n=1, p=net.l1.alpha.detach().cpu().numpy())
                g2 = np.random.binomial(n=1, p=net.l2.alpha.detach().cpu().numpy())
            

                gammas = np.concatenate((g1.flatten(), g2.flatten()))
                density[i] = gammas.mean() #compute density for each model in the ensemble

            output1 = outputs.mean(0)
            output2 = outputs2.mean(0)

            class_pred = output1.round().squeeze()
            class_pred2 = output2.round().squeeze()
         
        
            a = ((class_pred == target) * 1).sum().item()
            b = ((class_pred2 == target) * 1).sum().item()
         
    
            ensemble.append(a)
            median.append(b)


    metr.append(np.sum(ensemble) / TEST_SIZE)
    metr.append(np.sum(median) / TEST_SIZE)
    print(np.mean(density), 'density')
    metr.append(np.mean(density))
    print(np.sum(ensemble) / TEST_SIZE, 'ensemble')
    print(np.sum(median) / TEST_SIZE, 'median')
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
savepath = current_dir +'/results/pistachio_LBBNN_metrics.txt'
    
np.savetxt(savepath, metrics_several_runs, delimiter=',')    
m = np.array(metrics_several_runs)
np.savetxt(current_dir +'/results/lbbnn_avg_metrics.txt',m.mean(axis = 0),delimiter = ',')
    

print(m.min(axis = 0),m.mean(axis = 0),m.max(axis = 0))
