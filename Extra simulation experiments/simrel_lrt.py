




# matplotlib inline
import math
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable

import pandas as pd

from torch.optim.lr_scheduler import MultiStepLR
from utils import BayesianLinearLrt, train


np.random.seed(1)



DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
LOADER_KWARGS = {'num_workers': 1, 'pin_memory': True} if torch.cuda.is_available() else {}


if (torch.cuda.is_available()):
    print("GPUs are used!")
else:
    print("CPUs are used!")

# define the parameters
BATCH_SIZE = 200
epochs = 500

num_models = 20
all_tpr = np.zeros(shape = (30,num_models))
all_fpr = np.zeros_like(all_tpr)

for i in range(30):

    X = np.genfromtxt('X_train' + str(i)+'.csv', delimiter=',',skip_header = 1)
    Y = np.genfromtxt('Y_train' + str(i) +'.csv', delimiter=',',skip_header = 1)
    
    
    
    
    x = np.array(X)
    y = np.array(Y)
    #y =y / y.std()
    dtrain = torch.tensor(np.column_stack((x, y)), dtype=torch.float32)

    
    
    
    
    
    TRAIN_SIZE = len(dtrain)
    NUM_BATCHES = TRAIN_SIZE / BATCH_SIZE
    
    
    n, p = X.shape
    
    class BayesianNetwork(nn.Module):
        def __init__(self):
            super().__init__()
            # set the architecture
            self.l1 = BayesianLinearLrt(p, 1) #one neuron = logistic regression
            self.loss = nn.BCELoss(reduction='sum')
    
        def forward(self, x):
            return torch.sigmoid(self.l1(x))
    
        def kl(self):
            return self.l1.kl
    

    
    
    true_weights = np.zeros(50)
    predicted_alphas = np.zeros(shape=(num_models, 50))
    true_weights[0:10] = 1
    true_weights = np.array([true_weights, ] * num_models)
    
    for j in range(0, num_models):
        print('model',j)
        torch.manual_seed(j)
        net = BayesianNetwork().to(DEVICE)
        optimizer = optim.Adam(net.parameters(),lr = 0.01)
        scheduler = MultiStepLR(optimizer, milestones=[250], gamma=0.1)
        for epoch in range(epochs):
            
            print('epoch =', epoch)
            loss = train(net,dtrain, optimizer,BATCH_SIZE,p,NUM_BATCHES = NUM_BATCHES)
            scheduler.step()
        predicted_alphas[j] = net.l1.alpha_q.data.detach().cpu().numpy().squeeze()
        a = net.l1.alpha_q.data.detach().cpu().numpy().squeeze() 
        aa = np.round(a,0)
        tw = (true_weights != 0) * 1
        print((aa == tw).mean())
        
        
    
       
    
    
    
    pa = np.round(predicted_alphas, 0) #median probability model
    tw = (true_weights != 0) * 1
    print((pa == tw).mean(axis=1))
    print((pa == tw).mean(axis=0))
    

    

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
    


    all_tpr[i],all_fpr[i] = get_TPR_FPR(pa,tw[0])
   
  
np.savetxt('all_tpr_lrt_binary'  +'.txt',all_tpr, delimiter=',',fmt='%s')
np.savetxt('all_fpr_lrt_binary' +'.txt',all_fpr, delimiter=',',fmt='%s')
