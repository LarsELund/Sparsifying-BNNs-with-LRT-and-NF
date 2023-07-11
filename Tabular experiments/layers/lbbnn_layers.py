#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import math
import numpy as np
import torch.nn.functional as F

TEMPER_PRIOR = 0.001
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


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

        # configuration of the layer
        self.in_features = in_features
        self.out_features = out_features

        # weight parameters initialization
        self.weight_mu = nn.Parameter(torch.Tensor(out_features, in_features).uniform_(-0.01, 0.01))
        self.weight_rho = nn.Parameter(-9 + 0.1 * torch.randn(out_features,in_features))
        self.weight = Gaussian(self.weight_mu, self.weight_rho)

        # weight priors initialization
        self.weight_a = nn.Parameter(torch.Tensor(1).uniform_(1, 1.1))
        self.weight_b = nn.Parameter(torch.Tensor(1).uniform_(1, 1.1))
        self.weight_prior = GaussGamma(self.weight_a, self.weight_b)

        # model parameters
        self.lambdal = nn.Parameter(torch.Tensor(out_features, in_features).uniform_(-10, 10))
        self.gammas = torch.Tensor(out_features, in_features).uniform_(0.99, 1)
        self.alpha = torch.Tensor(out_features, in_features).uniform_(0.999, 0.9999)
        self.gamma = Bernoulli(self.alpha)

        # model priors
        self.pa = nn.Parameter(torch.Tensor(1).uniform_(1, 1.1))
        self.pb = nn.Parameter(torch.Tensor(1).uniform_(1, 1.1))
        self.gamma_prior = BetaBinomial(pa=self.pa, pb=self.pb)

        # bias (intercept) parameters
        self.bias_mu = nn.Parameter(torch.Tensor(out_features).uniform_(-0.2, 0.2))
        self.bias_rho = nn.Parameter(-9+ 1 * torch.randn(out_features))
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