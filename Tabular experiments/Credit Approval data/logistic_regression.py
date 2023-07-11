#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
import os


full_data = np.loadtxt('dat.txt',dtype = float,delimiter = ',')


log_reg_scores = np.zeros(10)

X = full_data[:,0:15]
y = full_data[:,-1]
skf = StratifiedKFold(n_splits=10,shuffle = True,random_state = 1)

for i, (train_index, test_index) in enumerate(skf.split(X, y)):
        
        train_dat,y_train = X[train_index],y[train_index]
        test_dat,y_test = X[test_index],y[test_index]
        
        m1,s1 = train_dat[:,1].mean(),train_dat[:,1].std()
        m2,s2 = train_dat[:,2].mean(),train_dat[:,2].std()
        m7,s7 = train_dat[:,7].mean(),train_dat[:,7].std()
        m13,s13 = train_dat[:,13].mean(),train_dat[:,13].std()
        m14,s14 = train_dat[:,14].mean(),train_dat[:,14].std()


        train_dat[:,1] = (train_dat[:,1] - m1) / s1
        train_dat[:,2] = (train_dat[:,2] - m2) / s2
        train_dat[:,7] = (train_dat[:,7] - m7) / s7
        train_dat[:,13] = (train_dat[:,13] - m13) / s13
        train_dat[:,14] = (train_dat[:,14] - m14) / s14

        test_dat[:,1] = (test_dat[:,1] - m1) / s1
        test_dat[:,2] = (test_dat[:,2] - m2) / s2
        test_dat[:,7] = (test_dat[:,7] - m7) / s7
        test_dat[:,13] = (test_dat[:,13] - m13) / s13
        test_dat[:,14] = (test_dat[:,14] - m14) / s14
        
        
        clf = LogisticRegression(random_state=i).fit(train_dat, y_train)
        log_reg_scores[i] = clf.score(test_dat,y_test)
    
        


#current_dir = os.getcwd()
#np.savetxt(current_dir +'/results/xgboost_avg_metrics.txt',a,delimiter = ',')
#np.savetxt(current_dir +'/results/rf_avg_metrics.txt',b,delimiter = ',')