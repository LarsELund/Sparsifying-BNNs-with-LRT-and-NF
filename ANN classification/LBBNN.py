#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import time
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from torchvision import datasets, transforms
from torchvision.utils import make_grid
from tqdm import tqdm, trange

# define the summary writer
# select the device
DEVICE = torch.device("cuda:3" if torch.cuda.is_available() else "cpu")
LOADER_KWARGS = {'num_workers': 1, 'pin_memory': True} if torch.cuda.is_available() else {}

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
TEMPER = 0.001
TEMPER_PRIOR = 0.001
epochs = 250
pepochs = 0

# define the data loaders
train_loader = torch.utils.data.DataLoader(
    datasets.MNIST(
        './mnist', train=True, download=True,
        transform=transforms.ToTensor()),
    batch_size=BATCH_SIZE, shuffle=True, **LOADER_KWARGS)
test_loader = torch.utils.data.DataLoader(
    datasets.MNIST(
        './mnist', train=False, download=True,
        transform=transforms.ToTensor()),
    batch_size=TEST_BATCH_SIZE, shuffle=False, **LOADER_KWARGS)



# for the OOD entropy
mnist_loader = torch.utils.data.DataLoader(
    datasets.FashionMNIST('./fmnist', train=False, download=True, transform=transforms.ToTensor()), batch_size=TEST_BATCH_SIZE, shuffle=False)
TRAIN_SIZE = len(train_loader.dataset)
TEST_SIZE = len(test_loader.dataset)
NUM_BATCHES = len(train_loader)
NUM_TEST_BATCHES = len(test_loader)

assert (TRAIN_SIZE % BATCH_SIZE) == 0
assert (TEST_SIZE % TEST_BATCH_SIZE) == 0

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
        # Variable((self.bern.sample(alpha.size()).to(DEVICE)<alpha.type(torch.cuda.FloatTensor)).type(torch.cuda.FloatTensor)).to(DEVICE)
        return gamma

    def sample(self):
        self.bern.sample()

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
    def __init__(self, in_features, out_features, layer_id):
        super().__init__()

        # configuration of the layer
        self.layer = layer_id
        self.in_features = in_features
        self.out_features = out_features

        # weight parameters initialization
        self.weight_mu = nn.Parameter(torch.Tensor(out_features, in_features).uniform_(-0.01, 0.01))
        self.weight_rho = nn.Parameter(-9 + 0.1 * torch.randn(out_features, in_features))
        self.weight = Gaussian(self.weight_mu, self.weight_rho)

        # weight priors initialization
        self.weight_a = nn.Parameter(torch.Tensor(1).uniform_(1, 1.1))
        self.weight_b = nn.Parameter(torch.Tensor(1).uniform_(1, 1.1))
        self.weight_prior = GaussGamma(self.weight_a, self.weight_b)

        # model parameters
        self.lambdal = nn.Parameter(torch.Tensor(out_features, in_features).uniform_(0, 1))
        self.gammas = torch.Tensor(out_features, in_features).uniform_(0.99, 1)
        self.alpha = torch.Tensor(out_features, in_features).uniform_(0.999, 0.9999)
        self.gamma = Bernoulli(self.alpha)

        # model priors
        self.pa = nn.Parameter(torch.Tensor(1).uniform_(1, 1.1))
        self.pb = nn.Parameter(torch.Tensor(1).uniform_(1, 1.1))
        self.gamma_prior = BetaBinomial(pa=self.pa, pb=self.pb)

        # bias (intercept) parameters
        self.bias_mu = nn.Parameter(torch.Tensor(out_features).uniform_(-0.2, 0.2))
        self.bias_rho = nn.Parameter(-9 + 1 * torch.randn(out_features))
        self.bias = Gaussian(self.bias_mu, self.bias_rho)

        # bias (intercept) priors
        self.bias_a = nn.Parameter(torch.Tensor(out_features).uniform_(1, 1.1))
        self.bias_b = nn.Parameter(torch.Tensor(out_features).uniform_(1, 1.1))
        self.bias_prior = GaussGamma(self.bias_a, self.bias_b)

        # scalars
        self.log_prior = 0
        self.log_variational_posterior = 0
        self.lagrangian = 0

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
        # calculate the kl
        if self.training or calculate_log_probs:

            self.alpha = 1 / (1 + torch.exp(-self.lambdal))
            self.log_prior = self.weight_prior.log_prob(weight, cgamma) \
                           + self.bias_prior.log_prob(bias,torch.ones_like(bias)) \
                           + self.gamma_prior.log_prob(cgamma, pa=self.pa, pb=self.pb)
            self.log_variational_posterior = self.weight.full_log_prob(input=weight,gamma=cgamma) \
                           + self.gamma.log_prob(cgamma) + self.bias.log_prob(bias)
        else:
            self.log_prior, self.log_variational_posterior, = 0,0

        return F.linear(input, weight, bias)


# deine the whole BNN
class BayesianNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        # set the architecture
        self.l1 = BayesianLinear(28 * 28, 400, 1)
        self.l2 = BayesianLinear(400, 600, 1)
        self.l3 = BayesianLinear(600, 10, 1)

    def forward(self, x, g1, g2, g3, sample=False, medimean=False):
        x = x.view(-1, 28 * 28)
        x = F.relu(self.l1.forward(x, g1, sample, medimean))
        x = F.relu(self.l2.forward(x, g2, sample, medimean))
        x = F.log_softmax((self.l3.forward(x, g3, sample, medimean)), dim=1)
        return x

    def log_prior(self):
        return self.l1.log_prior \
               + self.l2.log_prior \
               + self.l3.log_prior

    def log_variational_posterior(self):
        return self.l1.log_variational_posterior \
               + self.l2.log_variational_posterior \
               + self.l3.log_variational_posterior

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
            self.l3.alpha = 1 / (1 + torch.exp(-self.l3.lambdal))
            self.l3.gamma.alpha = self.l3.alpha

            # sample the model
            cgamma1 = self.l1.gamma.rsample().to(DEVICE)
            cgamma2 = self.l2.gamma.rsample().to(DEVICE)
            cgamma3 = self.l3.gamma.rsample().to(DEVICE)

            # get the results
            outputs[i] = self.forward(input, g1=cgamma1, g2=cgamma2, g3=cgamma3, sample=True, medimean=False)
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




# Stochastic Variational Inference iteration
def train(net, optimizer, epoch, i):
    net.train()
    print('epoch',epoch)
    for batch_idx, (data, target) in enumerate(tqdm(train_loader)):
        data, target = data.to(DEVICE), target.to(DEVICE)
        net.zero_grad()
        loss, log_prior, log_variational_posterior, negative_log_likelihood = net.sample_elbo(data, target)
        loss.backward()
        if COND_OPT:
            net.l1.weight_mu.grad = net.l1.weight_mu.grad * net.l1.gammas.data
            net.l2.weight_mu.grad = net.l2.weight_mu.grad * net.l2.gammas.data
            net.l3.weight_mu.grad = net.l3.weight_mu.grad * net.l3.gammas.data
        optimizer.step()

    print('loss',loss.item())
    print('log_prior',log_prior.item())
    print('log_variational_posterior',log_variational_posterior.item())
    print('nll',negative_log_likelihood.item())
    return  negative_log_likelihood.item(),loss.item()

def test_ensemble(net):
    net.eval()
    correct1 = 0
    correct2 = 0
    correct3 = 0
    correct4 = 0
    cases3 = 0
    cases4 = 0
    ctr = 0

    spars = np.zeros(TEST_SAMPLES)
    gt1 = np.zeros((400, 784))
    gt2 = np.zeros((600, 400))
    gt3 = np.zeros((10, 600))
    metr = []
    density = np.zeros(TEST_SAMPLES)
    ensemble = []
    posterior_mean = []
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(DEVICE), target.to(DEVICE)
            outputs = torch.zeros(TEST_SAMPLES, TEST_BATCH_SIZE, CLASSES).to(DEVICE)
            for i in range(TEST_SAMPLES):
                # get the inclusion probabilities for all layers
                net.l1.alpha = 1 / (1 + torch.exp(-net.l1.lambdal))
                net.l1.gamma.alpha = net.l1.alpha
                net.l2.alpha = 1 / (1 + torch.exp(-net.l2.lambdal))
                net.l2.gamma.alpha = net.l2.alpha
                net.l3.alpha = 1 / (1 + torch.exp(-net.l3.lambdal))
                net.l3.gamma.alpha = net.l3.alpha

                # sample the model
                g1 = net.l1.gamma.rsample().to(DEVICE)
                g2 = net.l2.gamma.rsample().to(DEVICE)
                g3 = net.l3.gamma.rsample().to(DEVICE)
                ctr += 1

                spars[i] = spars[i] + ((torch.sum(g1 > 0.5).cpu().detach().numpy() + torch.sum(
                    g2 > 0.5).cpu().detach().numpy() + torch.sum(g3 > 0.5).cpu().detach().numpy()) / (
                                                   10 * 600 + 400 * 600 + 400 * 784))
                gt1 = gt1 + (g1 > 0.5).cpu().numpy()
                gt2 = gt2 + (g2 > 0.5).cpu().numpy()
                gt3 = gt3 + (g3 > 0.5).cpu().numpy()
                outputs[i] = net.forward(data, sample=True, medimean=False, g1=net.l1.gamma.rsample(),
                                         g2=net.l2.gamma.rsample(), g3=net.l3.gamma.rsample())
                g5 = net.l1.gamma.rsample().to(DEVICE)
                g6 = net.l2.gamma.rsample().to(DEVICE)
                g7 = net.l3.gamma.rsample().to(DEVICE)
                gamms = torch.cat((g5.flatten(), g6.flatten(), g7.flatten()))
                density[i] = gamms.mean()


                if (i == 0):
                    mydata_means = sigmoid(outputs[i].detach().cpu().numpy())
                    for j in range(TEST_BATCH_SIZE):
                        mydata_means[j] /= np.sum(mydata_means[j])
                else:
                    tmp = sigmoid(outputs[i].detach().cpu().numpy())
                    for j in range(TEST_BATCH_SIZE):
                        tmp[j] /= np.sum(tmp[j])
                    # print(sum(tmp[j]))
                    mydata_means = mydata_means + tmp

            mydata_means /= TEST_SAMPLES
            np.savetxt('mnist_means_base' + '.txt', mydata_means, delimiter=',')
            d = data.reshape(1000, 28 * 28).cpu().numpy()
            np.savetxt('mnist_data_base' + '.txt', d, delimiter=',')

            mean_out = net(data, sample=False, medimean=False, g1=net.l1.gamma.rsample(),
                           g2=net.l2.gamma.rsample(), g3=net.l3.gamma.rsample()) #posterior mean

            output1 = outputs.mean(0)
            pred1 = output1.max(1, keepdim=True)[1]
            pr = mean_out.max(1, keepdim=True)[1]
            p = pred1.squeeze().cpu().numpy()
            a = pr.eq(target.view_as(pred1)).sum(dim=1).squeeze().cpu().numpy()
            b = pred1.eq(target.view_as(pred1)).sum().item()
            posterior_mean.append(a.sum())
            ensemble.append(b)
            np.savetxt('mnist_predictions_base' + '.txt', p, delimiter=',')
            np.savetxt('mnist_truth_base' + '.txt', target.cpu().numpy(), delimiter=',')

    ps = ((np.sum(gt1 > 0) + np.sum(gt2 > 0) + np.sum(gt3 > 0)) / (10 * 600 + 400 * 600 + 400 * 784)) / 10
    print(spars / ctr)
    print(ps)
    metr.append(np.sum(posterior_mean) / TEST_SIZE)
    metr.append(np.sum(ensemble) / TEST_SIZE)
    print(np.mean(density), 'density')
    metr.append(np.mean(density))
    print(np.sum(posterior_mean) / TEST_SIZE, 'posterior mean')
    print(np.sum(ensemble) / TEST_SIZE, 'ensemble')
    return metr


def cdf(x, plot=True, *args, **kwargs):
    x = sorted(x)
    y = np.arange(len(x)) / len(x)
    return plt.plot(x, y, *args, **kwargs) if plot else (x, y)

from scipy.special import expit

def sigmoid(x):
    return expit(x)


def outofsample(net,medimod = False):
    net.eval()
    correct = 0
    corrects = np.zeros(TEST_SAMPLES + 2, dtype=int)
    # gt1 = np.zeros((400,784))
    # gt2 = np.zeros((600,400))
    # gt3 = np.zeros((10,600))
    # ots = np.zeros(obj, dtype=int)
    # dts = np.zeros((obj,5,784))
    entropies = np.zeros(10)
    count = 0
    k = 0
    spars = np.zeros(TEST_SAMPLES)
    with torch.no_grad():
        for data, target in mnist_loader:
            data, target = data.to(DEVICE), target.to(DEVICE)
            outputs = torch.zeros(TEST_SAMPLES + 2, TEST_BATCH_SIZE, CLASSES).to(DEVICE)
            for i in range(TEST_SAMPLES):
                # print(i)
                if medimod:
                    outputs[i] = net.forward(data, sample=True, medimean=False,
                                             g1=(net.l1.alpha.data > 0.5).type(torch.cuda.FloatTensor).to(DEVICE),
                                             g2=(net.l2.alpha.data > 0.5).type(torch.cuda.FloatTensor).to(DEVICE),
                                             g3=(net.l3.alpha.data > 0.5).type(torch.cuda.FloatTensor).to(DEVICE))
                else:
                    outputs[i] = net.forward(data, g1=net.l1.gamma.rsample(), g2=net.l2.gamma.rsample(),
                                             g3=net.l3.gamma.rsample(), sample=True)

                if (i == 0):
                    mydata_means = sigmoid(outputs[i].detach().cpu().numpy())
                    for j in range(TEST_BATCH_SIZE):
                        mydata_means[j] /= np.sum(mydata_means[j])
                else:
                    tmp = sigmoid(outputs[i].detach().cpu().numpy())
                    for j in range(TEST_BATCH_SIZE):
                        tmp[j] /= np.sum(tmp[j])
                    # print(sum(tmp[j]))
                    mydata_means = mydata_means + tmp
            mydata_means /= TEST_SAMPLES
            # print(np.sum(mydata_means))
            for j in range(TEST_BATCH_SIZE):
                if k == 0 and j == 0:
                    entropies = -np.sum(mydata_means[j] * np.log(mydata_means[j]))
                else:
                    entropies = np.append(entropies, -np.sum(mydata_means[j] * np.log(mydata_means[j])))
            k += 1
            output = outputs[1:TEST_SAMPLES].mean(0)
            preds = outputs.max(2, keepdim=True)[1]
            pred = output.max(1, keepdim=True)[1]  # index of max log-probability
            corrects += preds.eq(target.view_as(pred)).sum(dim=1).squeeze().cpu().numpy()
            correct += pred.eq(target.view_as(pred)).sum().item()
    #cdf(entropies.flatten())
    return entropies.flatten()


print("Classes loaded")

import time


nll_several_runs = []
loss_several_runs = []
metrics_several_runs = []


# make inference on 10 networks
for i in range(0,10):
    print('network',i)
    torch.manual_seed(i)
    net = BayesianNetwork().to(DEVICE)
    optimizer = optim.Adam([
        {'params': net.l1.bias_mu, 'lr': 0.0001},
        {'params': net.l2.bias_mu, 'lr': 0.0001},
        {'params': net.l3.bias_mu, 'lr': 0.0001},
        {'params': net.l1.bias_rho, 'lr': 0.0001},
        {'params': net.l2.bias_rho, 'lr': 0.0001},
        {'params': net.l3.bias_rho, 'lr': 0.0001},
        {'params': net.l1.weight_mu, 'lr': 0.0001},
        {'params': net.l2.weight_mu, 'lr': 0.0001},
        {'params': net.l3.weight_mu, 'lr': 0.0001},
        {'params': net.l1.weight_rho, 'lr': 0.0001},
        {'params': net.l2.weight_rho, 'lr': 0.0001},
        {'params': net.l3.weight_rho, 'lr': 0.0001},
        {'params': net.l1.pa, 'lr': 0.001},
        {'params': net.l2.pa, 'lr': 0.001},
        {'params': net.l3.pa, 'lr': 0.001},
        {'params': net.l1.pb, 'lr': 0.001},
        {'params': net.l2.pb, 'lr': 0.001},
        {'params': net.l3.pb, 'lr': 0.001},
        {'params': net.l1.weight_a, 'lr': 0.00001},
        {'params': net.l2.weight_a, 'lr': 0.00001},
        {'params': net.l3.weight_a, 'lr': 0.00001},
        {'params': net.l1.weight_b, 'lr': 0.00001},
        {'params': net.l2.weight_b, 'lr': 0.00001},
        {'params': net.l3.weight_b, 'lr': 0.00001},
        {'params': net.l1.bias_a, 'lr': 0.00001},
        {'params': net.l2.bias_a, 'lr': 0.00001},
        {'params': net.l3.bias_a, 'lr': 0.00001},
        {'params': net.l1.bias_b, 'lr': 0.00001},
        {'params': net.l2.bias_b, 'lr': 0.00001},
        {'params': net.l3.bias_b, 'lr': 0.00001},
        {'params': net.l1.lambdal, 'lr': 0.1},
        {'params': net.l2.lambdal, 'lr': 0.1},
        {'params': net.l3.lambdal, 'lr': 0.1}
    ], lr=0.0001)
    all_nll = []
    all_loss = []
    t1 = time.time()
    for epoch in range(epochs):
        if (net.l1.pa / (net.l1.pa + net.l1.pb)).mean() < 0.1 or epoch == 20:
            print(epoch)
            net.l1.gamma_prior.exact = True
            net.l2.gamma_prior.exact = True
            net.l3.gamma_prior.exact = True
            net.l1.bias_prior.exact = True
            net.l2.bias_prior.exact = True
            net.l3.bias_prior.exact = True
            net.l1.weight_prior.exact = True
            net.l2.weight_prior.exact = True
            net.l3.weight_prior.exact = True
            optimizer = optim.Adam([
                {'params': net.l1.bias_mu, 'lr': 0.0001},
                {'params': net.l2.bias_mu, 'lr': 0.0001},
                {'params': net.l3.bias_mu, 'lr': 0.0001},
                {'params': net.l1.bias_rho, 'lr': 0.0001},
                {'params': net.l2.bias_rho, 'lr': 0.0001},
                {'params': net.l3.bias_rho, 'lr': 0.0001},
                {'params': net.l1.weight_mu, 'lr': 0.0001},
                {'params': net.l2.weight_mu, 'lr': 0.0001},
                {'params': net.l3.weight_mu, 'lr': 0.0001},
                {'params': net.l1.weight_rho, 'lr': 0.0001},
                {'params': net.l2.weight_rho, 'lr': 0.0001},
                {'params': net.l3.weight_rho, 'lr': 0.0001},
                {'params': net.l1.pa, 'lr': 0.00},
                {'params': net.l2.pa, 'lr': 0.00},
                {'params': net.l3.pa, 'lr': 0.00},
                {'params': net.l1.pb, 'lr': 0.00},
                {'params': net.l2.pb, 'lr': 0.00},
                {'params': net.l3.pb, 'lr': 0.00},
                {'params': net.l1.weight_a, 'lr': 0.00},
                {'params': net.l2.weight_a, 'lr': 0.00},
                {'params': net.l3.weight_a, 'lr': 0.00},
                {'params': net.l1.weight_b, 'lr': 0.00},
                {'params': net.l2.weight_b, 'lr': 0.00},
                {'params': net.l3.weight_b, 'lr': 0.00},
                {'params': net.l1.bias_a, 'lr': 0.00},
                {'params': net.l2.bias_a, 'lr': 0.00},
                {'params': net.l3.bias_a, 'lr': 0.00},
                {'params': net.l1.bias_b, 'lr': 0.00},
                {'params': net.l2.bias_b, 'lr': 0.00},
                {'params': net.l3.bias_b, 'lr': 0.00},
                {'params': net.l1.lambdal, 'lr': 0.0001},
                {'params': net.l2.lambdal, 'lr': 0.0001},
                {'params': net.l3.lambdal, 'lr': 0.0001}
            ], lr=0.0001)
        nll,loss = train(net, optimizer, epoch, i)
        all_nll.append(nll)
        all_loss.append(loss)
    nll_several_runs.append(all_nll)
    loss_several_runs.append(all_loss)
    t = round((time.time() - t1), 1)

    net.l1.alpha.data = 1 / (1 + torch.exp(-net.l1.lambdal.data))
    # (torch.clamp(self.l1.alpha.data,1e-8 , 1-1e-8))
    net.l2.alpha.data = 1 / (1 + torch.exp(-net.l2.lambdal.data))
    # (torch.clamp(self.l2.alpha.data,1e-8 , 1-1e-8))
    net.l3.alpha.data = 1 / (1 + torch.exp(-net.l3.lambdal.data))
    # (torch.clamp(self.l3.alpha.data,1e-8 , 1-1e-8))
    net.l1.gamma.alpha.data = 1 / (1 + torch.exp(-net.l1.lambdal.data))
    # (torch.clamp(self.l1.alpha.data,1e-8 , 1-1e-8))
    net.l2.gamma.alpha.data = 1 / (1 + torch.exp(-net.l2.lambdal.data))
    # (torch.clamp(self.l2.alpha.data,1e-8 , 1-1e-8))
    net.l3.gamma.alpha.data = 1 / (1 + torch.exp(-net.l3.lambdal.data))
    # (torch.clamp(self.l3.alpha.data,1e-8 , 1-1e-8))

    net.l1.gamma.exact = True
    net.l2.gamma.exact = True
    net.l3.gamma.exact = True

    metrics = test_ensemble(net)
    metrics.append(t / epochs)
    print(t/epochs , 'time')
    metrics_several_runs.append(metrics)

    os = (torch.sum(net.l1.alpha.data > 0.5).cpu().detach().numpy() + torch.sum(
        net.l2.alpha.data > 0.5).cpu().detach().numpy() + torch.sum(net.l3.alpha.data > 0.5).cpu().detach().numpy()) / (
                     10 * 600 + 400 * 600 + 400 * 784)
    if i == 9: #compute the OOD entropy
        enr = outofsample(net)
        np.savetxt('ENTROPY_MNIST_FMNIST' + '.txt', enr, delimiter=',')

np.savetxt('MNISTNOFLOWLOSSES' + '.txt', loss_several_runs, delimiter=',')
np.savetxt('MNISTNOFLOWNLL' + '.txt', nll_several_runs, delimiter=',')
np.savetxt('MNISTNOFLOWMETRICS' + '.txt', metrics_several_runs, delimiter=',')

  