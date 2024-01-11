import torch
import torch.nn as nn
import math
from flows import PropagateFlow
DEVICE = 'mps'
Z_FLOW_TYPE = 'IAF'
R_FLOW_TYPE = 'IAF'
BATCH_SIZE = 621

class BayesianLinear(nn.Module):
    def __init__(self, in_features, out_features, num_transforms):
        super().__init__()
        self.out_features = out_features
        self.in_features = in_features
        self.weight_mu = nn.Parameter(0.1 * torch.randn(out_features,in_features))
        self.weight_logvar= nn.Parameter(- 9 +0.1 * torch.randn(out_features,in_features))
        self.weight_sigma = torch.empty(size=self.weight_logvar.shape)
        # model variational parameters

        # weight priors = N(0,1)
        self.mu_prior = torch.zeros(out_features, device=DEVICE)
        self.sigma_prior = (self.mu_prior + 1.0).to(DEVICE)

        # bias variational parameters
        self.bias_mu = nn.Parameter(0.001 * torch.randn(out_features))
        self.bias_logvar = nn.Parameter(-9+ 0.001 * torch.randn(out_features))
        self.bias_sigma = torch.empty(self.bias_logvar.shape)

       
        
        
        
        
        
        # scalars
        self.kl = 0
    

    # forward path
    def forward(self, input, ensemble=False):
        self.weight_sigma = self.weight_logvar.exp()
        self.bias_sigma = self.bias_logvar.exp()
    
      
        if self.training or ensemble:
            e_w = self.weight_mu 
            var_w = self.weight_sigma ** 2
            mu = (torch.mm(input, e_w.T) + self.bias_mu)
            var = torch.mm(input ** 2, var_w.T) + self.bias_sigma ** 2
           
            eps = torch.randn(size=(var.size()), device=DEVICE)
            activations = mu + torch.sqrt(var) * eps
         
            
            
        if self.training:

                self.kl=  (torch.log(self.sigma_prior /activations.std(axis = 0)) - 0.5 + (activations.var(axis = 0)
                    + (activations.mean(axis = 0) - self.mu_prior) ** 2) / (
                             2 * self.sigma_prior ** 2)).sum()
                        
                        
                        
                
                
                

        return activations