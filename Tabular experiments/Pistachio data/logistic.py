#!/usr/bin/env python3
# -*- coding: utf-8 -*-



import pandas as pd
import numpy as np

from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression
import os


df = pd.read_excel('Pistachio_28_Features_Dataset.xlsx')
y = df['Class']
y = y.replace('Kirmizi_Pistachio',1)
y = y.replace('Siirt_Pistachio',0)
df.drop('Class', inplace=True, axis=1)
full_data = np.column_stack((df,y))
full_data = np.array(full_data,dtype =float)


p = full_data.shape[1] - 1



log_reg_scores = np.zeros(10)

X = full_data[:,0:p]
y = full_data[:,-1]
skf = StratifiedKFold(n_splits=10,shuffle = True,random_state = 1)

for i, (train_index, test_index) in enumerate(skf.split(X, y)):
        
        X_train,y_train = X[train_index],y[train_index]
        X_test,y_test = X[test_index],y[test_index]
        
        mu,std = X_train.mean(axis = 0), X_train.std(axis = 0)
        X_tr = (X_train - mu) / std
        X_te = (X_test  - mu) / std
        clf = LogisticRegression(random_state=i).fit(X_tr, y_train)
        log_reg_scores[i] = clf.score(X_te,y_test)