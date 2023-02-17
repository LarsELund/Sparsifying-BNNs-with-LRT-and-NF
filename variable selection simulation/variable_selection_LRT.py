#!/usr/bin/env python3
# -*- coding: utf-8 -*-


# matplotlib inline
import math
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from tensorboardX import SummaryWriter
import pandas as pd


np.random.seed(1)

writer = SummaryWriter()
sns.set()
sns.set_style("dark")
sns.set_palette("muted")
sns.set_color_codes("muted")

# select the device
DEVICE = torch.device("cuda:3" if torch.cuda.is_available() else "cpu")
LOADER_KWARGS = {'num_workers': 1, 'pin_memory': True} if torch.cuda.is_available() else {}


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
epochs = 500



# import the data
# taken from https://github.com/aliaksah/EMJMCMC2016/tree/master/examples/Simulated%20Logistic%20Data%20With%20Multiple%20Modes%20(Example%203)

x_df = pd.read_csv('sim3-X.csv', header=None)
y_df = pd.read_csv('sim3-Y.csv', header=None)
x = np.array(x_df)
y = np.array(y_df)
data = torch.tensor(np.column_stack((x, y)), dtype=torch.float32)

data_mean = data.mean(axis=0)[6:20]
data_std = data.std(axis=0)[6:20]
data[:, 6:20] = (data[:, 6:20] - data_mean) / data_std

dtrain = data
TRAIN_SIZE = len(dtrain)
NUM_BATCHES = TRAIN_SIZE / BATCH_SIZE


n, p = dtrain.shape


class BayesianLinear(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        # weight variational parameters
        self.weight_mu = nn.Parameter(torch.Tensor(out_features, in_features).uniform_(-0.01, 0.01))
        self.weight_rho = nn.Parameter(torch.Tensor(out_features, in_features).uniform_(-5, -4))
        self.weight_sigma = torch.empty(size = self.weight_rho.shape)

        # weight priors = N(0,1)
        self.mu_prior = torch.zeros(out_features, in_features, device=DEVICE) 
        self.sigma_prior = (self.mu_prior + 1.).to(DEVICE)

        # model variational parameters
        self.lambdal = nn.Parameter(torch.Tensor(out_features, in_features).uniform_(1.5, 2.5))
        self.alpha_q = torch.empty(size = self.lambdal.shape)

        # prior inclusion probability
        self.alpha_prior = (self.mu_prior + 0.25).to(DEVICE)

        # bias variational parameters
        self.bias_mu = nn.Parameter(torch.Tensor(out_features).uniform_(-0.01, 0.01))
        self.bias_rho = nn.Parameter(torch.Tensor(out_features).uniform_(-5, -4))
        self.bias_sigma = torch.empty(self.bias_rho.shape)

        # bias priors = N(0,1)
        self.bias_mu_prior = torch.zeros(out_features, device=DEVICE)
        self.bias_sigma_prior = (self.bias_mu_prior + 1.).to(DEVICE)

        # scalars
        self.kl = 0

    # forward path
    def forward(self, input, sample=False, calculate_log_probs=False):
        self.alpha_q = 1 / (1 + torch.exp(-self.lambdal))
        self.weight_sigma = torch.log1p(torch.exp(self.weight_rho))
        self.bias_sigma = torch.log1p(torch.exp(self.bias_rho))
        if self.training or sample:
            e_w = self.weight_mu * self.alpha_q
            var_w = self.alpha_q*(self.weight_sigma ** 2 + (1 - self.alpha_q) * self.weight_mu ** 2)
            e_b = torch.mm(input, e_w.T) + self.bias_mu
            var_b = torch.mm(input ** 2, var_w.T) + self.bias_sigma ** 2
            eps = torch.randn(size=(var_b.size()), device=DEVICE)
            activations = e_b + torch.sqrt(var_b) * eps

        else:  # posterior mean
            e_w = self.weight_mu * self.alpha_q
            e_b = torch.mm(input, e_w.T) + self.bias_mu
            activations = e_b

        if self.training or calculate_log_probs:



            kl_bias = (torch.log(self.bias_sigma_prior / self.bias_sigma) - 0.5 + (self.bias_sigma ** 2
                    + (self.bias_mu - self.bias_mu_prior) ** 2) / (
                               2 * self.bias_sigma_prior ** 2)).sum()

            kl_weight = (self.alpha_q * (torch.log(self.sigma_prior / self.weight_sigma)
                                         - 0.5 + torch.log(self.alpha_q / self.alpha_prior)
                                         + (self.weight_sigma ** 2 + (self.weight_mu - self.mu_prior) ** 2) / (
                                                     2 * self.sigma_prior ** 2))
                         + (1 - self.alpha_q) * torch.log((1 - self.alpha_q) / (1 - self.alpha_prior))).sum()

            self.kl = kl_bias + kl_weight
        else:
            self.kl = 0

        return activations







class BayesianNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        # set the architecture
        self.l1 = BayesianLinear(p - 1, 1) #one layer with one neuron i.e. logistic regression
        self.loss = nn.BCELoss(reduction='sum') #output is 0 or 1

    def forward(self, x, sample=False):
        x = self.l1(x, sample)
        x = torch.sigmoid(x)
        return x

    def kl(self):
        return self.l1.kl


def train(net, optimizer, batch_size=BATCH_SIZE):
    net.train()
    old_batch = 0
    accs = []
    for batch in range(int(np.ceil(dtrain.shape[0] / batch_size))):
        batch = (batch + 1)
        _x = dtrain[old_batch: batch_size * batch, 0:p - 1]
        _y = dtrain[old_batch: batch_size * batch, -1]
        old_batch = batch_size * batch
        target = Variable(_y).to(DEVICE)
        data = Variable(_x).to(DEVICE)
        net.zero_grad()
        outputs = net(data, sample=True)
        target = target.unsqueeze(1).float()
        negative_log_likelihood = net.loss(outputs, target)
        loss = negative_log_likelihood + net.kl() / NUM_BATCHES
        loss.backward()
        optimizer.step()
        pred = outputs.squeeze().detach().cpu().numpy()
        pred = np.round(pred, 0)
        acc = np.mean(pred == _y.detach().cpu().numpy())
        accs.append(acc)

    print('loss', loss.item())
    print('nll', negative_log_likelihood.item())
    print('accuracy =', np.mean(accs))
    return negative_log_likelihood.item(), loss.item()





print("Classes loaded")
k = 100

predicted_alphas = np.zeros(shape=(k, 20)) #store the results for the k models

true_weights = np.array([-4, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1.2, 0, 37.1, 0, 0, 50, - 0.00005, 10, 3, 0])

true_weights = np.array([true_weights, ] * k)
import time

t = time.time()

for i in range(0, k): #use k = 100 models for the paper
    print('model',i)
    torch.manual_seed(i)
    net = BayesianNetwork().to(DEVICE)
    optimizer = optim.Adam(net.parameters(),lr = 0.01)
    for epoch in range(epochs):
        nll, loss = train(net, optimizer)
        print('epoch =', epoch)
    predicted_alphas[i] = net.l1.alpha_q.data.detach().cpu().numpy().squeeze()

print('elapsed time = ',time.time() - t,' seconds')  

pa = np.round(predicted_alphas, 0) #median probability 
tw = (true_weights != 0) * 1
print((pa == tw).mean(axis=1))
print((pa == tw).mean(axis=0))

np.savetxt('predicted_alphas_LRT' +'.txt',pa, delimiter=',',fmt='%s')
np.savetxt('true_weights_LRT' +'.txt',tw, delimiter=',',fmt='%s')
np.savetxt('alphas_LRT' +'.txt',predicted_alphas, delimiter=',',fmt='%s') #without rounding to 0,1



true_w = tw[0]

def get_TPR_FPR(predicted_alphas,true_weights):
    tpr = []
    fpr = []

    for a in predicted_alphas:
        tp = a[(a  == true_weights)].sum()
        fn = true_weights[a == 0].sum()
        fp = a[true_weights == 0].sum()
        tn = (true_weights[a == 0] == 0).sum()
        fpr.append(fp / (fp + tn))
        tpr.append(tp / (tp + fn))
    
    return tpr,fpr


tpr,fpr = get_TPR_FPR(pa,true_w)
print('tpr =',np.mean(tpr),'fpr =',np.mean(fpr))
