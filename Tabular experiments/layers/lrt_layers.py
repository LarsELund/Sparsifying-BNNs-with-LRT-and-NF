#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import torch.nn.functional as F
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "mps")

class BayesianLinear(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        # weight variational parameters
        self.weight_mu = nn.Parameter(torch.Tensor(out_features, in_features).uniform_(-0.01, 0.01))
        self.weight_rho = nn.Parameter(-9 + 0.1* torch.randn(out_features,in_features))
        self.weight_sigma = torch.empty(size=self.weight_rho.shape)

        # weight priors = N(0,1)
        self.mu_prior = torch.zeros(out_features, in_features, device=DEVICE)
        self.sigma_prior = (self.mu_prior + 1).to(DEVICE)
        
        # model variational parameters
        self.lambdal = nn.Parameter(torch.Tensor(out_features, in_features).uniform_(-10, 10))
        self.alpha_q = torch.empty(size=self.lambdal.shape)

        # model priors = Bernoulli(0.10)
        self.alpha_prior = (self.mu_prior + 0.1).to(DEVICE)

     

        # bias variational parameters
        self.bias_mu = nn.Parameter(torch.Tensor(out_features).uniform_(-0.2, 0.2))
        self.bias_rho = nn.Parameter(-9+ 1. * torch.randn(out_features))
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
            var_w = self.alpha_q* self.weight_sigma ** 2 + \
            self.alpha_q * (1 - self.alpha_q) * self.weight_mu ** 2 
            e_b = torch.mm(input, e_w.T) + self.bias_mu
            var_b = torch.mm(input ** 2, var_w.T) + self.bias_sigma ** 2
            eps = torch.randn(size=(var_b.size()), device=DEVICE)
            activations = e_b + torch.sqrt(var_b) * eps
            
           
            

            

        if self.training or calculate_log_probs:

        

            kl_bias = (torch.log(self.bias_sigma_prior / self.bias_sigma) - 0.5 + (self.bias_sigma ** 2
                    + (self.bias_mu - self.bias_mu_prior) ** 2) / (
                               2 * self.bias_sigma_prior ** 2)).sum()

            kl_weight = (self.alpha_q * (torch.log(self.sigma_prior / self.weight_sigma)
                                         - 0.5 + torch.log(self.alpha_q / self.alpha_prior)
                                         + (self.weight_sigma ** 2 + (self.weight_mu  - self.mu_prior) ** 2) / (
                                                 2 * self.sigma_prior ** 2))
                         + (1 - self.alpha_q) * torch.log((1 - self.alpha_q) / (1 - self.alpha_prior))).sum()

            self.kl = kl_bias + kl_weight 
        else:
            self.kl = 0

        return activations