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
# select the device
DEVICE = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
LOADER_KWARGS = {'num_workers': 1, 'pin_memory': True} if torch.cuda.is_available() else {}
torch.backends.cudnn.deterministic = True # To avoid non-deterministic behavior of conv layers

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

# define the data loaders

train_loader = torch.utils.data.DataLoader(
    datasets.FashionMNIST(
        './fmnist', train=True, download=True,
        transform=transforms.ToTensor()),
    batch_size=BATCH_SIZE, shuffle=True, **LOADER_KWARGS)
test_loader = torch.utils.data.DataLoader(
  datasets.FashionMNIST(
        './fmnist', train=False, download=True,
        transform=transforms.ToTensor()),
    batch_size=TEST_BATCH_SIZE, shuffle=False, **LOADER_KWARGS)


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
    def __init__(self, in_features, out_features):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        # weight parameters
        self.weight_mu = nn.Parameter(torch.Tensor(out_features, in_features).uniform_(-0.2, 0.2))
        self.weight_rho = nn.Parameter(-9 + 0.1 * torch.randn(out_features,in_features))
        self.weight = Gaussian(self.weight_mu, self.weight_rho)
        # weight priors
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
    def forward(self, input, cgamma, sample=False, calculate_log_probs=False):
        # if sampling
        if self.training or sample:
            self.gammas = cgamma
            ws = self.weight.rsample()
            weight = cgamma * ws
            bias = self.bias.rsample()
    
      
        else:
            w = self.weight.rsample()
            bias = self.bias.rsample()
            weight = w * cgamma
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

   
class BayesianConv2d(nn.Module):
    def __init__(self,in_channels,out_channels, kernel_size):
        super().__init__()

        if type(kernel_size) == int:
            kernel = (kernel_size,kernel_size)
            
        elif type(kernel_size) == tuple:
            kernel = kernel_size
        # Weight parameters
        self.weight_mu = nn.Parameter(torch.Tensor(out_channels,in_channels,kernel[0],kernel[1]).uniform_(-0.2,0.2))
        self.weight_rho = nn.Parameter(-9 + 0.1 * torch.randn(out_channels, in_channels, kernel[0], kernel[1]))
        self.weight = Gaussian(self.weight_mu, self.weight_rho)
        # weight priors
        self.weight_a = nn.Parameter(torch.Tensor(1).uniform_(1, 1.1))
        self.weight_b = nn.Parameter(torch.Tensor(1).uniform_(1, 1.1))
        self.weight_prior = GaussGamma(self.weight_a, self.weight_b)

        # model parameters
        self.lambdal = nn.Parameter(torch.Tensor(out_channels,in_channels,kernel[0],kernel[1]).uniform_(0, 1))
        self.gammas = torch.Tensor(out_channels,in_channels,kernel[0],kernel[1]).uniform_(0.99, 1)
        self.alpha = torch.Tensor(out_channels,in_channels,kernel[0],kernel[1]).uniform_(0.999, 0.9999)
        self.gamma = Bernoulli(self.alpha)
        # model priors
        self.pa = nn.Parameter(torch.Tensor(1).uniform_(1, 1.1))
        self.pb = nn.Parameter(torch.Tensor(1).uniform_(1, 1.1))
        self.gamma_prior = BetaBinomial(pa=self.pa, pb=self.pb)
        #bias parameters
        self.bias_mu = nn.Parameter(torch.Tensor(out_channels).uniform_(-0.2, 0.2))
        self.bias_rho = nn.Parameter(- 9 + 1 * torch.randn(out_channels))
        self.bias = Gaussian(self.bias_mu, self.bias_rho)
        # bias (intercept) priors
        self.bias_a = nn.Parameter(torch.Tensor(out_channels).uniform_(1, 1.1))
        self.bias_b = nn.Parameter(torch.Tensor(out_channels).uniform_(1, 1.1))
        self.bias_prior = GaussGamma(self.bias_a, self.bias_b)

        # scalars
        self.log_prior = 0
        self.log_variational_posterior = 0
        self.lagrangian = 0

    # forward path
    def forward(self, input, cgamma, sample=False, calculate_log_probs=False):
        # if sampling
        if self.training or sample:
            self.gammas = cgamma
            weight = cgamma * self.weight.rsample()
            bias = self.bias.rsample()
    
        # if joint mean in the space of models and parameters (for a given alpha vector)
        else:
            w = self.weight.rsample()
            bias = self.bias.rsample()
            weight = w * cgamma
        # calculate the losses
    
        if self.training or calculate_log_probs:
            self.alpha = 1 / (1 + torch.exp(-self.lambdal))
            self.log_prior = self.weight_prior.log_prob(weight, cgamma) + self.bias_prior.log_prob(bias,
                                                                                                   torch.ones_like(
                                                                                                       bias)) + self.gamma_prior.log_prob(
                cgamma, pa=self.pa, pb=self.pb)
                                                                                                      
            self.log_variational_posterior = self.weight.full_log_prob(input=weight,
                                                                       gamma=cgamma) + self.gamma.log_prob(
                cgamma) + self.bias.log_prob(bias)
        else:
            self.log_prior, self.log_variational_posterior
        # propogate
        return F.conv2d(input,weight,bias)

    # deine the whole BNN    

class BCNN(nn.Module):
    def __init__(self):
        super().__init__()
        ## LeNet-5 architechture
        self.conv1 = BayesianConv2d(in_channels=1, out_channels=32, kernel_size=5)
        self.conv2 = BayesianConv2d(in_channels=32, out_channels=48, kernel_size=5)
        self.pool = nn.AvgPool2d(kernel_size=2)
        self.fc1 = BayesianLinear(768, 256)
        self.fc2 = BayesianLinear(256, 84)
        self.fc3 = BayesianLinear(84, 10)
        
    def forward(self, x,g1,g2,g3,g4,g5,sample = False):
        x = x.reshape(x.size(0), 1, 28, 28)
        x = F.relu(self.conv1(x,g1, sample))
        x = self.pool(x)
        x = F.relu(self.conv2(x,g2, sample))
        x = self.pool(x)
        x = x.view(x.size(0), -1)  # reshape it back into a vector and do some fully connected layers
        x = F.relu(self.fc1(x,g3, sample))
        x = F.relu(self.fc2(x, g4, sample))
        x = F.log_softmax(self.fc3(x,g5, sample), dim=1)
        return x

    def log_prior(self):
        return sum([layer.log_prior for layer in self.children() if layer is not self.pool])
     

    def log_variational_posterior(self):
        return sum([layer.log_variational_posterior for layer in self.children() if layer is not self.pool])

    # sample the marginal likelihood lower bound
    def sample_elbo(self, input, target, samples=SAMPLES):
        outputs = torch.zeros(samples, BATCH_SIZE, CLASSES).to(DEVICE)
        log_priors = torch.zeros(samples).to(DEVICE)
        log_variational_posteriors = torch.zeros(samples).to(DEVICE)
        negative_log_likelihoods = torch.zeros(samples).to(DEVICE)
        for i in range(samples):
            # get the inclusion probabilities for all layers
            for layer in self.children():
                if layer is not self.pool:
                    layer.alpha = 1 / (1 + torch.exp(-layer.lambdal))
                    layer.gamma.alpha = layer.alpha

            # sample the model
            cgamma1 = self.conv1.gamma.rsample().to(DEVICE)
            cgamma2 = self.conv2.gamma.rsample().to(DEVICE)
            cgamma3 = self.fc1.gamma.rsample().to(DEVICE)
            cgamma4 = self.fc2.gamma.rsample().to(DEVICE)
            cgamma5 = self.fc3.gamma.rsample().to(DEVICE)
            # get the results
            outputs[i] = self.forward(input, g1=cgamma1, g2=cgamma2, g3=cgamma3,g4 = cgamma4 ,g5 = cgamma5,sample=True)
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
    metr = []
    ensemble = []
    median = []
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(DEVICE), target.to(DEVICE)
            outputs = torch.zeros(TEST_SAMPLES, TEST_BATCH_SIZE, CLASSES).to(DEVICE)
            out2 = torch.zeros_like(outputs)
            for i in range(TEST_SAMPLES):
                # get the inclusion probabilities for all layers
                for layer in net.children():
                    if not isinstance(layer,nn.AvgPool2d):
                        layer.alpha = 1 / (1 + torch.exp(-layer.lambdal))
                        layer.gamma.alpha = layer.alpha
            

                outputs[i] = net.forward(data, sample=True, g1=net.conv1.gamma.rsample(),
                                         g2=net.conv2.gamma.rsample(), g3=net.fc1.gamma.rsample(),
                                         g4 = net.fc2.gamma.rsample(),g5 = net.fc3.gamma.rsample())
                
                out2[i] = net.forward(data, sample=False,
                                             g1=(net.conv1.alpha.data > 0.5).type(torch.cuda.FloatTensor).to(DEVICE),
                                             g2=(net.conv2.alpha.data > 0.5).type(torch.cuda.FloatTensor).to(DEVICE),
                                             g3=(net.fc1.alpha.data > 0.5).type(torch.cuda.FloatTensor).to(DEVICE),
                                             g4=(net.fc2.alpha.data > 0.5).type(torch.cuda.FloatTensor).to(DEVICE),
                                             g5=(net.fc3.alpha.data > 0.5).type(torch.cuda.FloatTensor).to(DEVICE))

         
            output1 = outputs.mean(0)
            output2 = out2.mean(0)
            pred1 = output1.max(1, keepdim=True)[1]
            pred2 = output2.max(1, keepdim=True)[1]
            a = pred2.eq(target.view_as(pred2)).sum().item()
            b = pred1.eq(target.view_as(pred1)).sum().item()
            median.append(a)
            ensemble.append(b)

    g1 = ((net.fc1.alpha.detach().cpu().numpy() > 0.5) * 1.)
    g2 = ((net.fc2.alpha.detach().cpu().numpy() > 0.5) * 1.)
    g3 = ((net.fc3.alpha.detach().cpu().numpy() > 0.5) * 1.)
    g4 = ((net.conv1.alpha.detach().cpu().numpy() > 0.5) * 1.)
    g5 = ((net.conv2.alpha.detach().cpu().numpy() > 0.5) * 1.)

    gs = np.concatenate((g1.flatten(), g2.flatten(), g3.flatten(), g4.flatten(), g5.flatten()))
    metr.append(np.sum(median) / TEST_SIZE)
    metr.append(np.sum(ensemble) / TEST_SIZE)
    print(np.mean(gs), 'sparsity')
    metr.append(np.mean(gs))
    print(np.sum(median) / TEST_SIZE, 'median')
    print(np.sum(ensemble) / TEST_SIZE, 'ensemble')
    return metr
    


from scipy.special import expit

def sigmoid(x):
    return expit(x)






print("Classes loaded")

nll_several_runs = []
loss_several_runs = []
metrics_several_runs = []


# make inference on 10 networks
for i in range(0, 10):
    print('on nextwork',i)
    torch.manual_seed(i)
    net = BCNN().to(DEVICE)
    optimizer = optim.Adam([
        {'params': net.conv1.bias_mu, 'lr': 0.0001},
        {'params': net.conv2.bias_mu, 'lr': 0.0001},
        {'params': net.fc1.bias_mu, 'lr': 0.0001},
        {'params':net.fc2.bias_mu,'lr':0.0001},
        {'params': net.fc3.bias_mu, 'lr': 0.0001},

        {'params': net.conv1.bias_rho, 'lr': 0.0001},
        {'params': net.conv2.bias_rho, 'lr': 0.0001},
        {'params': net.fc1.bias_rho, 'lr': 0.0001},
        {'params':net.fc2.bias_rho,'lr':0.0001},
        {'params': net.fc3.bias_rho, 'lr': 0.0001},
        {'params': net.conv1.weight_mu, 'lr': 0.0001},
        {'params': net.conv2.weight_mu, 'lr': 0.0001},
        {'params': net.fc1.weight_mu, 'lr': 0.0001},
        {'params':net.fc2.weight_mu,'lr':0.0001},
        {'params': net.fc3.weight_mu, 'lr': 0.0001},

        {'params': net.conv1.weight_rho, 'lr': 0.0001},
        {'params': net.conv2.weight_rho, 'lr': 0.0001},
        {'params': net.fc1.weight_rho, 'lr': 0.0001},
        {'params':net.fc2.weight_rho,'lr':0.0001},
        {'params': net.fc3.weight_rho, 'lr': 0.0001},

        {'params': net.conv1.pa, 'lr': 0.001},
        {'params': net.conv2.pa, 'lr': 0.001},
        {'params': net.fc1.pa, 'lr': 0.001},
        {'params':net.fc2.pa,'lr':0.001},
        {'params': net.fc3.pa, 'lr': 0.001},
        {'params': net.conv1.pb, 'lr': 0.001},
        {'params': net.conv2.pb, 'lr': 0.001},
        {'params': net.fc1.pb, 'lr': 0.001},
        {'params':net.fc2.pb,'lr':0.0001},
        {'params': net.fc3.pb, 'lr': 0.0001},
        {'params': net.conv1.weight_a, 'lr': 0.00001},
        {'params': net.conv2.weight_a, 'lr': 0.00001},
        {'params': net.fc1.weight_a, 'lr': 0.00001},
        {'params':net.fc2.weight_a,'lr':0.00001},
        {'params': net.fc3.weight_a, 'lr': 0.00001},

        {'params': net.conv1.weight_b, 'lr': 0.00001},
        {'params': net.conv2.weight_b, 'lr': 0.00001},
        {'params': net.fc1.weight_b, 'lr': 0.00001},
        {'params':net.fc2.weight_b,'lr':0.00001},
        {'params': net.fc3.weight_b, 'lr': 0.00001},

        {'params': net.conv1.bias_a, 'lr': 0.00001},
        {'params': net.conv2.bias_a, 'lr': 0.00001},
        {'params': net.fc1.bias_a, 'lr': 0.00001},
        {'params':net.fc2.bias_a,'lr':0.00001},
        {'params': net.fc3.bias_a, 'lr': 0.00001},

        {'params': net.conv1.bias_b, 'lr': 0.00001},
        {'params': net.conv2.bias_b, 'lr': 0.00001},
        {'params': net.fc1.bias_b, 'lr': 0.00001},
        {'params':net.fc2.bias_b,'lr':0.00001},
        {'params': net.fc3.bias_b, 'lr': 0.00001},

        {'params': net.conv1.lambdal, 'lr': 0.1},
        {'params': net.conv2.lambdal, 'lr': 0.1},
        {'params': net.fc1.lambdal, 'lr': 0.1},
        {'params':net.fc2.lambdal,'lr':0.1},
        {'params': net.fc3.lambdal, 'lr': 0.1},

    ], lr=0.0001)
    all_nll = []
    all_loss = []
    t1 = time.time()

    for epoch in range(epochs):
        if epoch == 20:
            for layer in net.children():
                if not isinstance(layer, nn.AvgPool2d):
                    layer.gamma_prior.exact = True
                    layer.bias_prior.exact = True
                    layer.weight_prior.exact = True
            optimizer = optim.Adam([
                {'params': net.conv1.bias_mu, 'lr': 0.0001},
                {'params': net.conv2.bias_mu, 'lr': 0.0001},
                {'params': net.fc1.bias_mu, 'lr': 0.0001},
                {'params':net.fc2.bias_mu,'lr':0.0001},
                {'params': net.fc3.bias_mu, 'lr': 0.0001},

                {'params': net.conv1.bias_rho, 'lr': 0.0001},
                {'params': net.conv2.bias_rho, 'lr': 0.0001},
                {'params': net.fc1.bias_rho, 'lr': 0.0001},
                {'params':net.fc2.bias_rho,'lr':0.0001},
                {'params': net.fc3.bias_rho, 'lr': 0.0001},

                {'params': net.conv1.weight_mu, 'lr': 0.0001},
                {'params': net.conv2.weight_mu, 'lr': 0.0001},
                {'params': net.fc1.weight_mu, 'lr': 0.0001},
                {'params':net.fc2.weight_mu,'lr':0.0001},
                {'params': net.fc3.weight_mu, 'lr': 0.0001},

                {'params': net.conv1.weight_rho, 'lr': 0.0001},
                {'params': net.conv2.weight_rho, 'lr': 0.0001},
                {'params': net.fc1.weight_rho, 'lr': 0.0001},
                {'params':net.fc2.weight_rho,'lr':0.0001},
                {'params': net.fc3.weight_rho, 'lr': 0.0001},

                {'params': net.conv1.pa, 'lr': 0.00},
                {'params': net.conv2.pa, 'lr': 0.00},
                {'params': net.fc1.pa, 'lr': 0.00},
                {'params':net.fc2.pa,'lr':0.00},
                {'params': net.fc3.pa, 'lr': 0.00},

                {'params': net.conv1.pb, 'lr': 0.00},
                {'params': net.conv2.pb, 'lr': 0.00},
                {'params': net.fc1.pb, 'lr': 0.00},
                 {'params':net.fc2.pb,'lr':0.00},
                {'params': net.fc3.pb, 'lr': 0.00},

                {'params': net.conv1.weight_a, 'lr': 0.00},
                {'params': net.conv2.weight_a, 'lr': 0.00},
                {'params': net.fc1.weight_a, 'lr': 0.00},
                {'params':net.fc2.weight_a,'lr':0.00},
                {'params': net.fc3.weight_a, 'lr': 0.00},

                {'params': net.conv1.weight_b, 'lr': 0.00},
                {'params': net.conv2.weight_b, 'lr': 0.00},
                {'params': net.fc1.weight_b, 'lr': 0.00},
                {'params':net.fc2.weight_b,'lr':0.00},
                {'params': net.fc3.weight_b, 'lr': 0.00},

                {'params': net.conv1.bias_a, 'lr': 0.00},
                {'params': net.conv2.bias_a, 'lr': 0.00},
                {'params': net.fc1.bias_a, 'lr': 0.00},
                {'params':net.fc2.bias_a,'lr':0.00},
                {'params': net.fc3.bias_a, 'lr': 0.00},

                {'params': net.conv1.bias_b, 'lr': 0.00},
                {'params': net.conv2.bias_b, 'lr': 0.00},
                {'params': net.fc1.bias_b, 'lr': 0.00},
                {'params':net.fc2.bias_b,'lr':0.00},
                {'params': net.fc3.bias_b, 'lr': 0.00},

                {'params': net.conv1.lambdal, 'lr': 0.0001},
                {'params': net.conv2.lambdal, 'lr': 0.0001},
                {'params': net.fc1.lambdal, 'lr': 0.0001},
                {'params':net.fc2.lambdal,'lr':0.0001},
                {'params': net.fc3.lambdal, 'lr': 0.0001},

            ], lr=0.0001)
        nll,loss = train(net, optimizer, epoch, i)
        all_nll.append(nll)
        all_loss.append(loss)
    for layer in net.children():
        if not isinstance(layer,nn.AvgPool2d):
            layer.alpha.data = 1 / (1 + torch.exp(-layer.lambdal.data))
            layer.gamma.alpha.data = 1 / (1 + torch.exp(-layer.lambdal.data))
  


    net.conv1.gamma.exact = True
    net.conv2.gamma.exact = True
    net.fc1.gamma.exact = True
    net.fc2.gamma.exact = True
    net.fc3.gamma.exact = True
    nll_several_runs.append(all_nll)
    loss_several_runs.append(all_loss)
    t = round((time.time() - t1), 1)
    metrics = test_ensemble(net)
    metrics.append(t / epochs)
    print(t/epochs , 'time')
    metrics_several_runs.append(metrics)


np.savetxt('FMNIST_CNN_BASELINE_LOSS_MEDIAN' + '.txt', loss_several_runs, delimiter=',')
np.savetxt('FMNIST_CNN_BASELINE_NLL_MEDIAN' + '.txt', nll_several_runs, delimiter=',')
np.savetxt('FMNIST_CNN_BASELINE_METRICS_MEDIAN' + '.txt', metrics_several_runs, delimiter=',')

