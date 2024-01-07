#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from flows import PropagateFlow
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "mps")


Z_FLOW_TYPE = 'IAF'
R_FLOW_TYPE = 'IAF'

class BayesianLinear(nn.Module):
    def __init__(self, in_features, out_features,num_transforms):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        # weight variational parameters
        self.weight_mu = nn.Parameter(torch.Tensor(out_features, in_features).uniform_(-0.01, 0.01))
        self.weight_rho = nn.Parameter(-9 + 0.1 * torch.randn(out_features,in_features))
        self.weight_sigma = torch.empty(size=self.weight_rho.shape)

        # weight priors = N(0,1)
        self.mu_prior = torch.zeros(out_features, in_features, device=DEVICE)
        self.sigma_prior = (self.mu_prior + 1).to(DEVICE)
        

     

        # bias variational parameters
        self.bias_mu = nn.Parameter(torch.Tensor(out_features).uniform_(-0.2, 0.2))
        self.bias_rho = nn.Parameter(-9+ 1. * torch.randn(out_features))
        self.bias_sigma = torch.empty(self.bias_rho.shape)

        # bias priors = N(0,1)
        self.bias_mu_prior = torch.zeros(out_features, device=DEVICE)
        self.bias_sigma_prior = (self.bias_mu_prior + 1.).to(DEVICE)
        
        self.q0_mean = nn.Parameter(1 * torch.randn(in_features))
        self.q0_log_var = nn.Parameter(-9 + 1 * torch.randn(in_features))

        # c b1 and b2 variational parameters, same shape as z
        self.c1 = nn.Parameter(1 * torch.randn(in_features))

        self.r0_b1 = nn.Parameter(1 * torch.randn(in_features))
        self.r0_b2 = nn.Parameter(1 * torch.randn(in_features))

        # define flows for z and r
        self.z_flow = PropagateFlow(Z_FLOW_TYPE, in_features, num_transforms)
        self.r_flow = PropagateFlow(R_FLOW_TYPE, in_features, num_transforms)
        # scalars
        self.kl = 0
        self.z = 0
        
        

      
        # scalars
        self.kl = 0
        
    def sample_z(self):
        q0_std = self.q0_log_var.exp().sqrt()
        epsilon_z = torch.randn_like(q0_std)
        self.z = self.q0_mean + q0_std * epsilon_z  # reparametrization trick
        zs, log_det_q = self.z_flow(self.z)
        return zs, log_det_q.squeeze()
     
      

    # forward path
    def forward(self, input, sample=False, calculate_log_probs=False):
        self.weight_sigma = torch.log1p(torch.exp(self.weight_rho))
        self.bias_sigma = torch.log1p(torch.exp(self.bias_rho))
        z_k,_ = self.sample_z()
        if self.training or sample:

            e_w = self.weight_mu * z_k
            var_w = self.weight_sigma **2
            e_b = torch.mm(input, e_w.T) + self.bias_mu
            var_b = torch.mm(input ** 2, var_w.T) + self.bias_sigma ** 2
            eps = torch.randn(size=(var_b.size()), device=DEVICE)
            activations = e_b + torch.sqrt(var_b) * eps
            
           

        return activations
    
    
    def kl_div(self):
        z2, log_det_q = self.sample_z()  # z_0 -> z_k
        W_mean = z2 * self.weight_mu 
        W_var =self.weight_sigma**2
        log_q0 = (-0.5 * torch.log(torch.tensor(math.pi)) - 0.5 * self.q0_log_var
                  - 0.5 * ((self.z - self.q0_mean) ** 2 / self.q0_log_var.exp())).sum()
        log_q = -log_det_q + log_q0
        act_mu = self.c1 @ W_mean.T
        act_var = self.c1 ** 2 @ W_var.T
        act_inner = act_mu + act_var.sqrt() * torch.randn_like(act_var)
        a = nn.Hardtanh()
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
                    
        kl_weight = (torch.log(self.sigma_prior / self.weight_sigma) - 0.5 + (self.weight_sigma ** 2
                + (self.weight_mu *z2 - self.mu_prior) ** 2) / (
                           2 * self.sigma_prior ** 2)).sum()
        

       

        return kl_bias + kl_weight + log_q - log_r
    
    