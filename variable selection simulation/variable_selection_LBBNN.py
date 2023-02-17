#!/usr/bin/env python3
# -*- coding: utf-8 -*-


#matplotlib inline
import math
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
import pandas as pd
from torch.optim.lr_scheduler import ExponentialLR, MultiStepLR

# select the device
DEVICE = torch.device("cuda:3" if torch.cuda.is_available() else "cpu")
LOADER_KWARGS = {'num_workers': 1, 'pin_memory': True} if torch.cuda.is_available() else {}

np.random.seed(1)


if (torch.cuda.is_available()):
    print("GPUs are used!")
else:
    print("CPUs are used!")

# define the parameters
BATCH_SIZE = 400
TEST_BATCH_SIZE = 10
COND_OPT = False
CLASSES = 2
SAMPLES = 1
TEST_SAMPLES = 10
TEMPER = 0.001
TEMPER_PRIOR = 0.001
epochs = 500
#import the data

x_df = pd.read_csv('sim3-X.csv',header = None)
y_df = pd.read_csv('sim3-Y.csv',header = None)
x = np.array(x_df)
y = np.array(y_df)
data = torch.tensor(np.column_stack((x,y)),dtype = torch.float32)


means = data[:,6:19].mean(axis = 0)
std = data[:, 6:19].std(axis = 0)

data[:, 6:19] = (data[:,6:19] - means) / std



dtrain = data
TRAIN_SIZE = len(dtrain)
#TEST_SIZE = len(te_ids)
NUM_BATCHES = TRAIN_SIZE/BATCH_SIZE
#NUM_TEST_BATCHES = len(te_ids)/BATCH_SIZE


n,p = dtrain.shape

# define Gaussian distribution
class Gaussian(object):
    def __init__(self, mu, rho):
        super().__init__()
        self.mu = mu
        self.rho = rho
        self.normal = torch.distributions.Normal(0, 1)

    @property
    def sigma(self):
        return torch.log1p(torch.exp(self.rho))
        #return torch.exp(self.rho)

    def rsample(self):
        epsilon = self.normal.sample(self.rho.size()).to(DEVICE)
        return self.mu + self.sigma * epsilon

    def log_prob(self, input):
        return (-math.log(math.sqrt(2 * math.pi))
                - torch.log(self.sigma)
                - ((input - self.mu) ** 2) / (2 * self.sigma ** 2)).sum()

    def log_prob_iid(self, input):
        return (-math.log(math.sqrt(2 * math.pi))
                - torch.log(self.sigma)
                - ((input - self.mu) ** 2) / (2 * self.sigma ** 2))

    def full_log_prob(self, input, gamma):
        return (torch.log(gamma * (torch.exp(self.log_prob_iid(input)))
                          + (1 - gamma) + 1e-8)).sum()


# define Bernoulli distribution
class Bernoulli(object):
    def __init__(self, alpha):
        super().__init__()
        self.alpha = alpha
        self.exact = False

    def rsample(self):
        if self.exact:
            gamma = torch.distributions.Bernoulli(self.alpha).sample().to(DEVICE)
        else:
            gamma = torch.distributions.RelaxedBernoulli(probs=self.alpha, temperature=TEMPER_PRIOR).rsample()
        return gamma

    def sample(self):
        return torch.distributions.Bernoulli(self.alpha).sample().to(DEVICE)

    def log_prob(self, input):
        if self.exact:
            gamma = torch.round(input.detach())
            output = (gamma * torch.log(self.alpha + 1e-8) + (1 - gamma) * torch.log(1 - self.alpha + 1e-8)).sum()
        else:
            output = (input * torch.log(self.alpha + 1e-8) + (1 - input) * torch.log(1 - self.alpha + 1e-8)).sum()
        return output


# define Normal-Gamma distribution
class GaussGamma(object):
    def __init__(self, a, b):
        super().__init__()
        self.a = a
        self.b = b
        self.exact = False
        self.sigma = torch.distributions.Gamma(self.a, self.b)

    def log_prob(self, input, gamma):
        tau = self.sigma.rsample()
        if self.exact:
            gamma1 = torch.round(gamma.detach())
            output = (gamma1 * (self.a * torch.log(self.b) + (self.a - 0.5) * tau - self.b * tau - torch.lgamma(
                self.a) - 0.5 * torch.log(torch.tensor(2 * np.pi))) - tau * torch.pow(input, 2) + (
                                  1 - gamma1) + 1e-8).sum()
        else:
            output = (gamma * (self.a * torch.log(self.b) + (self.a - 0.5) * tau - self.b * tau - torch.lgamma(
                self.a) - 0.5 * torch.log(torch.tensor(2 * np.pi))) - tau * torch.pow(input, 2) + (
                                  1 - gamma) + 1e-8).sum()
        return output


# define BetaBinomial distibution
class BetaBinomial(object):
    def __init__(self, pa, pb):
        super().__init__()
        self.pa = pa
        self.pb = pb
        self.exact = False

    def log_prob(self, input, pa, pb):
        if self.exact:
            gamma = torch.round(input.detach())
        else:
            gamma = input
        return (torch.lgamma(torch.ones_like(input)) + torch.lgamma(gamma + torch.ones_like(input) * self.pa)
                + torch.lgamma(torch.ones_like(input) * (1 + self.pb) - gamma) + torch.lgamma(
                    torch.ones_like(input) * (self.pa + self.pb))
                - torch.lgamma(torch.ones_like(input) * self.pa + gamma)
                - torch.lgamma(torch.ones_like(input) * 2 - gamma) - torch.lgamma(
                    torch.ones_like(input) * (1 + self.pa + self.pb))
                - torch.lgamma(torch.ones_like(input) * self.pa) - torch.lgamma(torch.ones_like(input) * self.pb)).sum()

    def rsample(self):
        gamma = torch.distributions.RelaxedBernoulli(
            probs=torch.distributions.Beta(self.pa, self.pb).rsample().to(DEVICE), temperature=0.001).rsample().to(
            DEVICE)
        return gamma

# define the linear layer for the BNN
class BayesianLinear(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()

        # configuration of the layer
        self.in_features = in_features
        self.out_features = out_features

        # weight parameters
        self.weight_mu = nn.Parameter(torch.Tensor(out_features, in_features).uniform_(-0.01, 0.01))
        self.weight_rho = nn.Parameter(torch.Tensor(out_features, in_features).uniform_(-5, -4))
        self.weight = Gaussian(self.weight_mu, self.weight_rho)
        # weight priors
        self.weight_a = nn.Parameter(torch.Tensor(1).uniform_(1, 1.1))
        self.weight_b = nn.Parameter(torch.Tensor(1).uniform_(1, 1.1))
        self.weight_prior = GaussGamma(self.weight_a, self.weight_b)

        # model parameters
        self.lambdal = nn.Parameter(torch.Tensor(out_features, in_features).uniform_(-0.5, 0.5))
        self.gammas = torch.Tensor(out_features, in_features).uniform_(0.99, 1)
        self.alpha = torch.Tensor(out_features, in_features).uniform_(0.999, 0.9999)
        self.gamma = Bernoulli(self.alpha)
        # model priors
        self.pa = nn.Parameter(torch.Tensor(1).uniform_(1, 1.1))
        self.pb = nn.Parameter(torch.Tensor(1).uniform_(1, 1.1))
        self.gamma_prior = BetaBinomial(pa=self.pa, pb=self.pb)

        # bias (intercept) parameters
        self.bias_mu = nn.Parameter(torch.Tensor(out_features).uniform_(-0.2, 0.2))
        self.bias_rho = nn.Parameter(torch.Tensor(out_features).uniform_(-5, -4))
        self.bias = Gaussian(self.bias_mu, self.bias_rho)
        # bias (intercept) priors
        self.bias_a = nn.Parameter(torch.Tensor(out_features).uniform_(1, 1.1))
        self.bias_b = nn.Parameter(torch.Tensor(out_features).uniform_(1, 1.1))
        self.bias_prior = GaussGamma(self.bias_a, self.bias_b)

        # scalars
        self.log_prior = 0
        self.log_variational_posterior = 0

    # forward path
    def forward(self, input, cgamma, sample=False, medimean=False, calculate_log_probs=False):
        # if sampling
        if self.training or sample:
            self.gammas = cgamma
            ws = self.weight.rsample()
            weight = cgamma * ws
            bias = self.bias.rsample()
        # if mean of the given model (e.g.) median probability model
        elif medimean:
            weight = cgamma * self.weight.mu
            bias = self.bias.mu
        # if joint mean in the space of models and parameters (for a given alpha vector)
        else:
            weight = self.alpha * self.weight.mu
            bias = self.bias.mu
        # calculate the losses
        if self.training or calculate_log_probs:
        
            self.alpha = 1 / (1 + torch.exp(-self.lambdal))
            self.log_prior = self.weight_prior.log_prob(ws, cgamma) + self.bias_prior.log_prob(bias,
                                                                                                   torch.ones_like(
                                                                                                       bias)) + self.gamma_prior.log_prob(
                cgamma, pa=self.pa, pb=self.pb)
            self.log_variational_posterior = self.weight.full_log_prob(input=ws,
                                                                       gamma=cgamma) + self.gamma.log_prob(
                cgamma) + self.bias.log_prob(bias)
        else:
            self.log_prior, self.log_variational_posterior, = 0,0
        # propogate
        
        return F.linear(input, weight, bias)

        # deine the whole BNN


# deine the whole BNN
class BayesianNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        # set the architecture
        self.l1 = BayesianLinear(20,1)

    def forward(self, x, g1, sample=False, medimean=False):
        x = self.l1(x,g1, sample, medimean)
        x = torch.sigmoid(x)
        return x

    def log_prior(self):
        return self.l1.log_prior 
             #  + self.l2.log_prior \
              # + self.l3.log_prior

    def log_variational_posterior(self):
        return self.l1.log_variational_posterior 
            #   + self.l2.log_variational_posterior \
            #   + self.l3.log_variational_posterior

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
            cgamma1 = self.l1.gamma.rsample().to(DEVICE)
            # get the results
            outputs[i] = self(input, g1=cgamma1, sample=True, medimean=False)
            log_priors[i] = self.log_prior()
            log_variational_posteriors[i] = self.log_variational_posterior()
            loss  = nn.BCELoss(reduction = 'sum')
            target = target.unsqueeze(1).float()
            #print(outputs[i])
            negative_log_likelihoods[i] = loss(outputs[i], target)
        # the current log prior
        log_prior = log_priors.mean()
        # the current log variational posterior
        log_variational_posterior = log_variational_posteriors.mean()
        # the current negative log likelihood
        negative_log_likelihood = negative_log_likelihoods.mean()
        out = outputs.mean(dim = 0)
        # the current ELBO
        loss = negative_log_likelihood  + (log_variational_posterior - log_prior) / NUM_BATCHES
        return loss, log_prior, log_variational_posterior, negative_log_likelihood,out


from scipy.special import expit

def sigmoid(x):
    return expit(x)


# Stochastic Variational Inference iteration
def train(net, optimizer, epoch, i, batch_size = BATCH_SIZE):
    net.train()
    old_batch = 0

    accs = []
    for batch in range(int(np.ceil(dtrain.shape[0] / batch_size))):
        
        batch = (batch + 1)
        _x = dtrain[old_batch: batch_size * batch,0:p-1]
        _y = dtrain[old_batch: batch_size * batch,-1]
        old_batch = batch_size * batch
        target = Variable(_y).to(DEVICE)
        data = Variable(_x).to(DEVICE)
        net.zero_grad()
        loss, log_prior, log_variational_posterior, negative_log_likelihood,out = net.sample_elbo(data, target)
        loss.backward()
        optimizer.step()
        pred = out.squeeze().detach().cpu().numpy()
        pred = np.round(pred,0) # positive class if > 0.5
        acc = np.mean(pred == _y.detach().cpu().numpy())
        accs.append(acc)
        
    print('loss',loss.item())
    print('log_prior',log_prior.item())
    print('log_variational_posterior',log_variational_posterior.item())
    print('nll',negative_log_likelihood.item())
    print('accuracy =',np.mean(accs))
    return  negative_log_likelihood.item(),loss.item()




print("Classes loaded")

k = 100 # number of times to run the model
predicted_alphas = np.zeros(shape = (k,20))

true_alphas = np.array([0.97,0.36,0.40,0.88,0.46,0.29,1.,0.31,0.61,0.44,
                           0.91,0.35,1.,0.44,0.35,1.,1.,1.,1.,0.37])

true_weights = np.array([-4.,0.,1.,0.,0.,0.,1.,0.,0.,0.,1.2,0.,37.1,0.,0.,50.,-0.00005,10.,3.,0.])

true_alphas = np.array([true_alphas,]*k)
true_weights = np.array([true_weights,]*k)

for i in range(0, k):
    print(i)
    torch.manual_seed(i)
    net = BayesianNetwork().to(DEVICE)
    optimizer = optim.SGD([
        {'params': net.l1.bias_mu, 'lr': 0.0001},
        {'params': net.l1.bias_rho, 'lr': 0.0001},
        {'params': net.l1.weight_mu, 'lr': 0.0001},
        {'params': net.l1.weight_rho, 'lr': 0.0001},
        {'params': net.l1.pa, 'lr': 0.001},
        {'params': net.l1.pb, 'lr': 0.001},
        {'params': net.l1.weight_a, 'lr': 0.001},
        {'params': net.l1.weight_b, 'lr': 0.001},
        {'params': net.l1.bias_a, 'lr': 0.001},
      
        {'params': net.l1.bias_b, 'lr': 0.001},
      
        {'params': net.l1.lambdal, 'lr': 0.001},
       
    ], lr=0.01)
    for epoch in range(epochs):
        print('epoch =',epoch)
        if (net.l1.pa / (net.l1.pa + net.l1.pb)).mean() < 0.1 or epoch == 50:
            print(epoch)
            net.l1.gamma_prior.exact = True
           
            net.l1.bias_prior.exact = True
         
            net.l1.weight_prior.exact = True
    
            optimizer = optim.SGD([
                {'params': net.l1.bias_mu, 'lr': 0.0001},
                
                {'params': net.l1.bias_rho, 'lr': 0.0001},
              
                {'params': net.l1.weight_mu, 'lr': 0.0001},
              
                {'params': net.l1.weight_rho, 'lr': 0.0001},
               
                {'params': net.l1.pa, 'lr': 0.00},
              
                {'params': net.l1.pb, 'lr': 0.00},
              
                {'params': net.l1.weight_a, 'lr': 0.00},
               
                {'params': net.l1.weight_b, 'lr': 0.00},
               
                {'params': net.l1.bias_a, 'lr': 0.00},
              
                {'params': net.l1.bias_b, 'lr': 0.00},
              
                {'params': net.l1.lambdal, 'lr': 0.001},
                
            ], lr=0.001)
            
        
        nll,loss = train(net, optimizer, epoch, i)
    net.l1.alpha = 1 / (1 + torch.exp(-net.l1.lambdal))
    predicted_alphas[i] = net.l1.alpha.data.detach().cpu().numpy().squeeze()
  


pa = np.round(predicted_alphas,0) 
tw = (true_weights != 0) * 1
print((pa == tw).mean(axis = 1))
print((pa == tw).mean(axis = 0))

np.savetxt('predicted_alphas_LBNNN' +'.txt',pa, delimiter=',',fmt='%s')
