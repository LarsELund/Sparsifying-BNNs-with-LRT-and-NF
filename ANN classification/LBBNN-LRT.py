#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from tqdm import tqdm
from torch.optim.lr_scheduler import MultiStepLR

# select the device
DEVICE = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
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
NUM_TEST_BATCHES = len(test_loader)

assert (TRAIN_SIZE % BATCH_SIZE) == 0
assert (TEST_SIZE % TEST_BATCH_SIZE) == 0


class BayesianLinear(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        # weight variational parameters
        self.weight_mu = nn.Parameter(torch.Tensor(out_features, in_features).uniform_(-0.2, 0.2))
        self.weight_rho = nn.Parameter(-9 + 0.1 * torch.randn(out_features, in_features))
        self.weight_sigma = torch.empty(size=self.weight_rho.shape)

        # weight priors = N(0,1)
        self.mu_prior = torch.zeros(out_features, in_features, device=DEVICE)
        self.sigma_prior = (self.mu_prior + 1.).to(DEVICE)

        # model variational parameters
        self.lambdal = nn.Parameter(torch.Tensor(out_features, in_features).uniform_(-15, 10))
        self.alpha_q = torch.empty(size=self.lambdal.shape)

        # model priors = Bernoulli(0.05)
        self.alpha_prior = (self.mu_prior + prior_inclusion_prob).to(DEVICE)

        # bias variational parameters
        self.bias_mu = nn.Parameter(torch.Tensor(out_features).uniform_(-0.2, 0.2))
        self.bias_rho = nn.Parameter(-9 + 1 * torch.randn(out_features))
        self.bias_sigma = torch.empty(self.bias_rho.shape)

        # bias priors = N(0,1)
        self.bias_mu_prior = torch.zeros(out_features, device=DEVICE)
        self.bias_sigma_prior = (self.bias_mu_prior + 1.).to(DEVICE)

        # scalars
        self.kl = 0

    # forward path
    def forward(self, input, ensemble=False, calculate_log_probs=False):
        self.alpha_q = 1 / (1 + torch.exp(-self.lambdal))
        self.weight_sigma = torch.log1p(torch.exp(self.weight_rho))
        self.bias_sigma = torch.log1p(torch.exp(self.bias_rho))
        if self.training or ensemble:
            e_w = self.weight_mu * self.alpha_q
            var_w = self.alpha_q * (self.weight_sigma ** 2 + (1 - self.alpha_q) * self.weight_mu ** 2)
            e_b = torch.mm(input, e_w.T) + self.bias_mu
            var_b = torch.mm(input ** 2, var_w.T) + self.bias_sigma ** 2
            eps = torch.randn(size=(var_b.size()), device=DEVICE)
            activations = e_b + torch.sqrt(var_b) * eps

        else:  # median prob
            w = torch.normal(self.weight_mu, self.weight_sigma)
            b = torch.normal(self.bias_mu, self.bias_sigma)
            g = (self.alpha_q.detach() > 0.5) * 1.
            weight = w * g
            activations = torch.matmul(input, weight.T) + b

        if self.training or calculate_log_probs:

            kl_bias = (torch.log(self.bias_sigma_prior / self.bias_sigma) - 0.5 + (self.bias_sigma ** 2
                                                                                   + (
                                                                                               self.bias_mu - self.bias_mu_prior) ** 2) / (
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
        super().__init__()  # define the architecture of the network, same as in baseline method
        self.l1 = BayesianLinear(28 * 28, 400)
        self.l2 = BayesianLinear(400, 600)
        self.l3 = BayesianLinear(600, 10)

    def forward(self, x, ensemble=False):  # forward propagation from 28x28 img input to softmax output
        x = x.view(-1, 28 * 28)
        x = F.relu(self.l1.forward(x, ensemble))
        x = F.relu(self.l2.forward(x, ensemble))
        x = F.log_softmax((self.l3.forward(x, ensemble)), dim=1)
        return x

    def kl(self):  # sum up kl divergence for the three layers
        return self.l1.kl + self.l2.kl + self.l3.kl


# Stochastic Variational Inference iteration
def train(net, optimizer):
    net.train()
    for batch_idx, (data, target) in enumerate(tqdm(train_loader)):
        data, target = data.to(DEVICE), target.to(DEVICE)
        net.zero_grad()
        outputs = net(data, ensemble=True)
        negative_log_likelihood = F.nll_loss(outputs, target, reduction='sum')  # compute nll between y_hat and y
        ELBO = negative_log_likelihood + net.kl() / NUM_BATCHES  # the approximate negative ELBO (one sample)
        ELBO.backward()
        optimizer.step()
    print('ELBO', ELBO.item())
    print('nll', negative_log_likelihood.item())
    return negative_log_likelihood.item(), ELBO.item()


def test_ensemble(net):
    net.eval()
    metr = []
    ensemble = []
    median = []

    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(DEVICE), target.to(DEVICE)
            outputs = torch.zeros(TEST_SAMPLES, TEST_BATCH_SIZE, CLASSES).to(DEVICE)
            out2 = torch.zeros_like(outputs)
            for i in range(TEST_SAMPLES):
                outputs[i] = net(data, ensemble=True)  # model avg over structures and weights
                out2[i] = net(data, ensemble=False)  # only model avg over weights where a > 0.5

            output1 = outputs.mean(0)
            out2 = out2.mean(0)

            pred1 = output1.max(1, keepdim=True)[1]  # index of max log-probability
            pred2 = out2.max(1, keepdim=True)[1]

            a = pred2.eq(target.view_as(pred2)).sum().item()
            b = pred1.eq(target.view_as(pred1)).sum().item()
            median.append(a)
            ensemble.append(b)
    # get the sparsity estimate
    g1 = ((net.l1.alpha_q.detach().cpu().numpy() > 0.5) * 1.)
    g2 = ((net.l2.alpha_q.detach().cpu().numpy() > 0.5) * 1.)
    g3 = ((net.l3.alpha_q.detach().cpu().numpy() > 0.5) * 1.)
    gs = np.concatenate((g1.flatten(), g2.flatten(), g3.flatten()))
    metr.append(np.sum(median) / TEST_SIZE)
    metr.append(np.sum(ensemble) / TEST_SIZE)
    metr.append(np.mean(gs))
    print(np.mean(gs), 'sparsity')
    print(np.sum(median) / TEST_SIZE, 'median')
    print(np.sum(ensemble) / TEST_SIZE, 'ensemble')
    return metr


import time

print("Classes loaded")

nll_several_runs = []
loss_several_runs = []
metrics_several_runs = []

for i in range(0, 10):
    print('network', i)
    torch.manual_seed(i)
    net = BayesianNetwork().to(DEVICE)
    optimizer = optim.Adam(net.parameters(), lr=0.0001)
    scheduler = MultiStepLR(optimizer, milestones=[125], gamma=0.1)
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

np.savetxt('KMNIST_ACT_LOSS' + '.txt', loss_several_runs, delimiter=',', fmt='%s')
np.savetxt('KMNIST_ACT_METRICS' + '.txt', metrics_several_runs, delimiter=',', fmt='%s')
np.savetxt('KMNIST_ACT_NLL' + '.txt', nll_several_runs, delimiter=',', fmt='%s')
