#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
import pandas as pd
import os

full_data = pd.read_csv('bank_data.csv', sep = ",",skiprows = 8)
full_data = np.array(full_data)
boost_scores = np.zeros(10)
forest_scores = np.zeros(10)
p = full_data.shape[1] -1

X = full_data[:,0:p]
y = full_data[:,-1]
skf = StratifiedKFold(n_splits=10,shuffle = True,random_state = 1)

log_reg_scores = np.zeros(10)

X = full_data[:,0:15]
y = full_data[:,-1]
skf = StratifiedKFold(n_splits=10,shuffle = True,random_state = 1)

for i, (train_index, test_index) in enumerate(skf.split(X, y)):
        
        train_dat,y_train = X[train_index],y[train_index]
        test_dat,y_test = X[test_index],y[test_index]
        
       
        
        
        clf = LogisticRegression(random_state=i).fit(train_dat, y_train)
        log_reg_scores[i] = clf.score(test_dat,y_test)