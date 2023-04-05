#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from tqdm import tqdm, trange
from torch.optim.lr_scheduler import MultiStepLR

# select the device
DEVICE = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")
LOADER_KWARGS = {'num_workers': 1, 'pin_memory': True} if torch.cuda.is_available() else {}
torch.backends.cudnn.deterministic = True  # To avoid non-deterministic behavior of conv layers

if (torch.cuda.is_available()):
    print("GPUs are used!")
else:
    print("CPUs are used!")

# define the parameters
BATCH_SIZE = 100
TEST_BATCH_SIZE = 1000
COND_OPT = False
CLASSES = 10
SAMPLES = 1
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
        self.lambdal = nn.Parameter(torch.Tensor(out_features, in_features).uniform_(-10, 15))
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

        else:  #median prob
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


class BayesianConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size):
        super().__init__()

        if type(kernel_size) == int:
            kernel = (kernel_size, kernel_size)

        elif type(kernel_size) == tuple:
            kernel = kernel_size

        # weight parameters
        self.weight_mu = nn.Parameter(torch.Tensor(out_channels, in_channels, kernel[0], kernel[1]).uniform_(-0.2, 0.2))
        self.weight_rho = nn.Parameter(-9 + 0.1 * torch.randn(out_channels, in_channels, kernel[0], kernel[1]))
        self.weight_sigma = torch.empty(self.weight_rho.shape)

        # weight priors
        self.mu_prior = torch.zeros((out_channels, in_channels, kernel[0], kernel[1]), device=DEVICE)
        self.sigma_prior = (self.mu_prior + 1.).to(DEVICE)

        # model parameters
        self.lambdal = nn.Parameter(torch.Tensor(out_channels, in_channels, kernel[0], kernel[1]).uniform_(-10, 15))
        self.alpha_q = torch.empty(self.lambdal.shape)

        # model prior
        self.alpha_prior = (self.mu_prior + prior_inclusion_prob).to(DEVICE)

        # bias parameters
        self.bias_mu = nn.Parameter(torch.Tensor(out_channels).uniform_(-0.2, 0.2))
        self.bias_rho = nn.Parameter(-9 + 1. * torch.randn(out_channels))
        self.bias_sigma = torch.empty(self.bias_rho.shape)

        # bias prior
        self.bias_mu_prior = torch.zeros(out_channels, device=DEVICE)
        self.bias_sigma_prior = (self.bias_mu_prior + 1.).to(DEVICE)

        # scalars
        self.kl = 0

    def forward(self, input, ensemble=False, calculate_log_probs=False):
        self.alpha_q = 1 / (1 + torch.exp(-self.lambdal))
        self.weight_sigma = torch.log1p(torch.exp(self.weight_rho))
        self.bias_sigma = torch.log1p(torch.exp(self.bias_rho))
        if self.training or ensemble:
            w_mean = self.weight_mu * self.alpha_q
            w_var = self.alpha_q * (self.weight_sigma ** 2 + (1 - self.alpha_q) * self.weight_mu ** 2)
            psi = F.conv2d(input, weight=w_mean, bias=self.bias_mu)
            delta = F.conv2d(input ** 2, weight=w_var, bias=self.bias_sigma ** 2)
            zeta = torch.randn_like(delta)
            activations = psi + torch.sqrt(delta) * zeta

        else:
            w = torch.normal(self.weight_mu,self.weight_sigma)
            bias = torch.normal(self.bias_mu,self.bias_sigma)
            g = (self.alpha_q.detach() > 0.5) * 1.
            weight = w * g
            activations = F.conv2d(input, weight=weight, bias=bias)
        

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
        # propogate
        return activations

    # deine the whole BNN


class BCNN(nn.Module):
    def __init__(self):
        super().__init__()
        # LeNet-5 architechture (sort of)
        self.conv1 = BayesianConv2d(in_channels=1, out_channels=32, kernel_size=5)
        self.conv2 = BayesianConv2d(in_channels=32, out_channels=48, kernel_size=5)
        self.pool = nn.AvgPool2d(kernel_size=2)
        self.fc1 = BayesianLinear(768, 256)
        self.fc2 = BayesianLinear(256, 84)
        self.fc3 = BayesianLinear(84, 10)


    def forward(self, x, ensemble=False):
        x = x.reshape(x.size(0), 1, 28, 28)
        x = F.relu(self.conv1(x, ensemble))
        x = self.pool(x)
        x = F.relu(self.conv2(x, ensemble))
        x = self.pool(x)
        x = x.view(x.size(0), -1)  # reshape it back into a vector and do some fully connected layers
        x = F.relu(self.fc1(x, ensemble))
        x = F.relu(self.fc2(x, ensemble))
        x = F.log_softmax(self.fc3(x, ensemble), dim=1)
        return x

    def kl(self):
        return self.conv1.kl + self.conv2.kl + self.fc1.kl + self.fc2.kl + self.fc3.kl


def train(net, optimizer):
    net.train()
    for batch_idx, (data, target) in enumerate(tqdm(train_loader)):
        data, target = data.to(DEVICE), target.to(DEVICE)
        net.zero_grad()
        outputs = net(data, ensemble=True)
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
    ensemble = []
    median = []
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(DEVICE), target.to(DEVICE)
            outputs = torch.zeros(TEST_SAMPLES, TEST_BATCH_SIZE, CLASSES).to(DEVICE)
            out2 = torch.zeros_like(outputs)
            for i in range(TEST_SAMPLES):
                outputs[i] = net(data, ensemble =True)
                out2[i] = net(data, ensemble=False)

            output1 = outputs.mean(0)
            out2 = out2.mean(0)
            pred1 = output1.max(1, keepdim=True)[1]
            pred2 = out2.max(1, keepdim=True)[1] 
            a = pred2.eq(target.view_as(pred2)).sum().item()
            b = pred1.eq(target.view_as(pred1)).sum().item()
            median.append(a)
            ensemble.append(b)

    g1 = ((net.fc1.alpha_q.detach().cpu().numpy() > 0.5) * 1.)
    g2 = ((net.fc2.alpha_q.detach().cpu().numpy() > 0.5) * 1.)
    g3 = ((net.fc3.alpha_q.detach().cpu().numpy() > 0.5) * 1.)
    g4 = ((net.conv1.alpha_q.detach().cpu().numpy() > 0.5) * 1.)
    g5 = ((net.conv2.alpha_q.detach().cpu().numpy() > 0.5) * 1.)

    gs = np.concatenate((g1.flatten(), g2.flatten(), g3.flatten(), g4.flatten(), g5.flatten()))


    metr.append(np.sum(median) / TEST_SIZE)
    metr.append(np.sum(ensemble) / TEST_SIZE)
    print(np.mean(gs), 'sparsity')
    metr.append(np.mean(gs))
    print(np.sum(median) / TEST_SIZE,'median')
    print(np.sum(ensemble) / TEST_SIZE, 'ensemble')
    return metr

import time

print("Classes loaded")

nll_several_runs = []
loss_several_runs = []
metrics_several_runs = []

# make inference on 10 networks
for i in range(0, 10):
    print('network', i)
    torch.manual_seed(i)
    net = BCNN().to(DEVICE)
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

np.savetxt('KMNIST_CNN_LRT_LOSS_MEDIAN' + '.txt', loss_several_runs, delimiter=',', fmt='%s')
np.savetxt('KMNIST_CNN_LRT_METRICS_MEDIAN' +'.txt', metrics_several_runs, delimiter=',',fmt='%s')
np.savetxt('KMNIST_CNN_LRT_NLL_MEDIAN' + '.txt', nll_several_runs, delimiter=',', fmt='%s')