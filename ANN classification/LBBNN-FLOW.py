#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import MultiStepLR
from torchvision import datasets, transforms
from tqdm import tqdm, trange
from flows_classification import PropagateFlow


# select the device
DEVICE = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")
LOADER_KWARGS = {'num_workers': 1, 'pin_memory': True} if torch.cuda.is_available() else {}

if (torch.cuda.is_available()):
    print("GPUs are used!")
else:
    print("CPUs are used!")

# define the parameters
BATCH_SIZE = 100
TEST_BATCH_SIZE = 1000
CLASSES = 10
TEST_SAMPLES = 100
epochs = 250
prior_inclusion_prob = 0.10
Z_FLOW_TYPE = 'IAF'
R_FLOW_TYPE = 'IAF'
# define the data loaders
train_loader = torch.utils.data.DataLoader(
    datasets.KMNIST(
        './kmnist', train=True, download=True,
        transform=transforms.ToTensor()),
    batch_size=BATCH_SIZE, shuffle=True, **LOADER_KWARGS)
test_loader = torch.utils.data.DataLoader(
    datasets.KMNIST(
        './kmnist', train=False, download=True,
        transform=transforms.ToTensor()),
    batch_size=TEST_BATCH_SIZE, shuffle=False, **LOADER_KWARGS)


TRAIN_SIZE = len(train_loader.dataset)
TEST_SIZE = len(test_loader.dataset)
NUM_BATCHES = len(train_loader)
assert (TRAIN_SIZE % BATCH_SIZE) == 0
assert (TEST_SIZE % TEST_BATCH_SIZE) == 0


class BayesianLinear(nn.Module):
    def __init__(self, in_features, out_features, num_transforms):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        # weight variational parameters
        self.weight_mu = nn.Parameter(torch.Tensor(out_features, in_features).uniform_(-0.01, 0.01))
        self.weight_rho = nn.Parameter(-9 + 0.1 * torch.randn(out_features,in_features))
        self.weight_sigma = torch.empty(size = self.weight_rho.shape)

        # prior distribution on all weights is N(0,1) 
        self.mu_prior = torch.zeros(out_features, in_features, device=DEVICE)
        self.sigma_prior = (self.mu_prior + 1.).to(DEVICE)

        # initialize the posterior inclusion probability. Here we must have alpha_q in (0,1)
        self.lambdal = nn.Parameter(torch.Tensor(out_features, in_features).uniform_(2.5,3.0))
        self.alpha_q = torch.empty(size = self.lambdal.shape)

        # prior inclusion probability. Used 0.05 for the experiments
        self.alpha_prior = (self.mu_prior + prior_inclusion_prob).to(DEVICE)

        # initialize the bias parameter

        self.bias_mu = nn.Parameter(torch.Tensor(out_features).uniform_(-0.2, 0.2))
        self.bias_rho = nn.Parameter(-9 + 1 * torch.randn(out_features))
        self.bias_sigma = torch.empty(self.bias_rho.shape)
        
         # bias priors = N(0,1)
        self.bias_mu_prior = torch.zeros(out_features, device=DEVICE)
        self.bias_sigma_prior = (self.bias_mu_prior + 1.).to(DEVICE)

        # z variational parameters
        self.q0_mean = nn.Parameter(1 * torch.randn(in_features) )
        self.q0_log_var = nn.Parameter(-9 + 1 * torch.randn(in_features))

        # c b1 and b2 variational parameters, same shape as z
        self.c1 = nn.Parameter(1 * torch.randn(in_features))

       
        self.r0_b1 = nn.Parameter(1* torch.randn(in_features))
        self.r0_b2 = nn.Parameter(1* torch.randn(in_features))

        # define flows for z and r
        self.z_flow = PropagateFlow(Z_FLOW_TYPE, in_features, num_transforms)
        self.r_flow = PropagateFlow(R_FLOW_TYPE, in_features, num_transforms)
        # scalars
        self.kl = 0
        self.z = 0

    def sample_z(self):
        q0_std = self.q0_log_var.exp().sqrt()
        epsilon_z = torch.randn_like(q0_std)
        self.z = self.q0_mean + q0_std * epsilon_z  # reparametrization trick
        zs, log_det_q = self.z_flow(self.z)
        return zs, log_det_q.squeeze()
    
    def kl_div(self):
        z2, log_det_q = self.sample_z()  # z_0 -> z_k
        W_mean = z2 * self.weight_mu * self.alpha_q
        W_var = self.alpha_q*(self.weight_sigma ** 2 + (1 - self.alpha_q) * self.weight_mu ** 2 *z2**2)
        log_q0 = (-0.5 * torch.log(torch.tensor(math.pi)) - 0.5 * self.q0_log_var
                  - 0.5 * ((self.z - self.q0_mean) ** 2 / self.q0_log_var.exp())).sum()
        log_q = -log_det_q + log_q0
        act_mu = self.c1 @ W_mean.T
        act_var = self.c1 ** 2 @ W_var.T
        act_inner = act_mu + act_var.sqrt() * torch.randn_like(act_var)
        a = nn.LeakyReLU(0.001)
        act = a(act_inner)
        mean_r = self.r0_b1.outer(act).mean(-1)  # eq (9) from MNF paper
        log_var_r = self.r0_b2.outer(act).mean(-1)  # eq (10) from MNF paper
        z_b, log_det_r = self.r_flow(z2)  # z_k - > z_b
        log_rb = (-0.5 * torch.log(torch.tensor(math.pi)) - 0.5 * log_var_r
                    - 0.5 * ((z_b[-1] - mean_r) ** 2 / log_var_r.exp())).sum()
        log_r = log_det_r + log_rb

        kl_bias = (torch.log(self.bias_sigma_prior / self.bias_sigma) - 0.5 + (self.bias_sigma ** 2
                + (self.bias_mu - self.bias_mu_prior) ** 2) / (
                2 * self.bias_sigma_prior ** 2)).sum()

        kl_weight = (self.alpha_q * (torch.log(self.sigma_prior / self.weight_sigma)
                  - 0.5 + torch.log(self.alpha_q / self.alpha_prior)
                  + (self.weight_sigma ** 2 + (self.weight_mu * z2 - self.mu_prior) ** 2) / (2 * self.sigma_prior ** 2))
                  + (1 - self.alpha_q) * torch.log((1 - self.alpha_q) / (1 - self.alpha_prior))).sum()

        return kl_bias + kl_weight + log_q - log_r
        

    # forward path
    def forward(self, input, sample=False):
        self.alpha_q = 1 / (1 + torch.exp(-self.lambdal))
        self.weight_sigma = torch.log1p(torch.exp(self.weight_rho))
        self.bias_sigma = torch.log1p(torch.exp(self.bias_rho))
        z_k, _ = self.sample_z()

        if self.training or sample:
            e_w = self.weight_mu * self.alpha_q * z_k
            var_w = self.alpha_q*(self.weight_sigma ** 2 + (1 - self.alpha_q) * self.weight_mu ** 2 *z_k**2)
            e_b = torch.mm(input, e_w.T) + self.bias_mu
            var_b = torch.mm(input ** 2, var_w.T) + self.bias_sigma ** 2
            eps = torch.randn(size=(var_b.size()), device=DEVICE)
            activations = e_b + torch.sqrt(var_b) * eps
        
        else:  # posterior mea
            e_w = self.weight_mu * self.alpha_q * z_k
            e_b = torch.mm(input, e_w.T) + self.bias_mu
            activations = e_b

        return activations


class BayesianNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.l1 = BayesianLinear(28 * 28, 400, num_transforms=2)
        self.l2 = BayesianLinear(400, 600, num_transforms=2)
        self.l3 = BayesianLinear(600, 10, num_transforms=2)
     
    def forward(self, x, sample=False):
        x = x.view(-1, 28 * 28)
        x = F.relu(self.l1.forward(x, sample))
        x = F.relu(self.l2.forward(x, sample))
        x = F.log_softmax((self.l3.forward(x, sample)), dim=1)
        return x

    def kl(self):
        return self.l1.kl_div() + self.l2.kl_div() + self.l3.kl_div()


# Stochastic Variational Inference iteration
def train(net, optimizer):
    net.train()
    for batch_idx, (data, target) in enumerate(tqdm(train_loader)):
        data, target = data.to(DEVICE), target.to(DEVICE)
        net.zero_grad()
        outputs = net(data, sample=True)
        negative_log_likelihood = F.nll_loss(outputs, target, reduction='sum')
        loss = negative_log_likelihood + net.kl() / NUM_BATCHES
        loss.backward()
        optimizer.step()
    print('loss', loss.item())
    print('nll', negative_log_likelihood.item())
    return negative_log_likelihood.item(), loss.item()


def test_ensemble(net):
    net.eval()
    metr = []
    density = np.zeros(TEST_SAMPLES)
    ensemble = []
    posterior_mean = []
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(DEVICE), target.to(DEVICE)
            outputs = torch.zeros(TEST_SAMPLES, TEST_BATCH_SIZE, CLASSES).to(DEVICE)
            for i in range(TEST_SAMPLES):
                outputs[i] = net(data, sample=True)

                ## sample the inclusion variables for each layer to estimate the sparsity level
                g1 = np.random.binomial(n=1, p=net.l1.alpha_q.detach().cpu().numpy())
                g2 = np.random.binomial(n=1, p=net.l2.alpha_q.detach().cpu().numpy())
                g3 = np.random.binomial(n=1, p=net.l3.alpha_q.detach().cpu().numpy())

                gammas = np.concatenate((g1.flatten(), g2.flatten(), g3.flatten()))
                density[i] = gammas.mean() #compute density for each model in the ensemble

            output1 = outputs.mean(0)
            mean_out = net(data, sample=False)
            pr = mean_out.max(1, keepdim=True)[1]
            pred1 = output1.max(1, keepdim=True)[1]  # index of max log-probability

            a = pr.eq(target.view_as(pred1)).sum(dim=1).squeeze().cpu().numpy()
            b = pred1.eq(target.view_as(pred1)).sum().item()
            posterior_mean.append(a.sum())
            ensemble.append(b)

    metr.append(np.sum(posterior_mean) / TEST_SIZE)
    metr.append(np.sum(ensemble) / TEST_SIZE)
    print(np.mean(density), 'density')
    metr.append(np.mean(density))
    print(np.sum(posterior_mean) / TEST_SIZE,'posterior mean')
    print(np.sum(ensemble) / TEST_SIZE, 'ensemble')
    return metr

import time

nll_several_runs = []
loss_several_runs = []
metrics_several_runs = []

# make inference on 10 networks
for i in range(0, 10):
    print('network', i)
    torch.manual_seed(i)
    net = BayesianNetwork().to(DEVICE)
    optimizer = optim.Adam(net.parameters(), lr=0.0001)
    scheduler = MultiStepLR(optimizer, milestones=[80,160], gamma=0.1)
    all_nll = []
    all_loss = []
    t1 = time.time()
    for epoch in range(epochs):
        print('epoch', epoch)
        nll, loss = train(net, optimizer)
        scheduler.step()
        all_nll.append(nll)
        all_loss.append(loss)
    nll_several_runs.append(all_nll)
    loss_several_runs.append(all_loss)
    t = round((time.time() - t1), 1)
    metrics = test_ensemble(net)
    metrics.append(t / epochs)
    metrics_several_runs.append(metrics)

np.savetxt('KMNIST_KL_loss_FLOW' + '.txt', loss_several_runs, delimiter=',',fmt='%s')
np.savetxt('KMNIST_KL_metrics_FLOW' + '.txt', metrics_several_runs, delimiter=',',fmt='%s')
np.savetxt('KMNIST_KL_nll_FLOW' + '.txt', nll_several_runs, delimiter=',',fmt='%s')