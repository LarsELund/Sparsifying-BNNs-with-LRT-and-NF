#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression

df = pd.read_excel('Dry_Bean_Dataset.xlsx',skiprows = 11) #jremove 11 samples

y = df.iloc[:,16]
y = y.replace('SEKER',0)
y = y.replace('BARBUNYA',1)
y = y.replace('BOMBAY',2)
y = y.replace('CALI',3)
y = y.replace('DERMASON',4)
y = y.replace('HOROZ',5)
y = y.replace('SIRA',6)
df.drop(df.columns[16], inplace=True, axis=1)
full_data = np.column_stack((df,y))


p = full_data.shape[1] - 1

skf = StratifiedKFold(n_splits=10,shuffle = True,random_state = 1)

log_reg_scores = np.zeros(10)

X = full_data[:,0:p]
y = full_data[:,-1]
skf = StratifiedKFold(n_splits=10,shuffle = True,random_state = 1)

for i, (train_index, test_index) in enumerate(skf.split(X, y)):
        
        train_dat,y_train = X[train_index],y[train_index]
        test_dat,y_test = X[test_index],y[test_index]
        
        tr = (train_dat - train_dat.mean(axis = 0)) / train_dat.std(axis = 0)
        te = (test_dat - train_dat.mean(axis = 0)) / train_dat.std(axis = 0)
        
        
        clf = LogisticRegression(random_state=i).fit(tr, y_train)
        log_reg_scores[i] = clf.score(te,y_test)