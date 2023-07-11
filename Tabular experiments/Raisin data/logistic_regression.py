import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
import os


df = pd.read_excel('Raisin_Dataset.xlsx')
y = df['Class']
y = y.replace('Kecimen',1)
y = y.replace('Besni',0)
df.drop('Class', inplace=True, axis=1)
full_data = np.column_stack((df,y))


p = full_data.shape[1] - 1

log = np.zeros(10)

X = full_data[:,0:p]
y = full_data[:,-1]
skf = StratifiedKFold(n_splits=10,shuffle = True,random_state = 1)

for i, (train_index, test_index) in enumerate(skf.split(X, y)):
        
        train_dat,y_train = X[train_index],y[train_index]
        test_dat,y_test = X[test_index],y[test_index]
        mu,std = train_dat.mean(axis = 0), train_dat.std(axis = 0)
        X_tr = (train_dat - mu) / std
        X_te = (test_dat  - mu) / std
        
       
        
        
        clf = LogisticRegression(random_state=i).fit(X_tr, y_train)
        log[i] = clf.score(X_te,y_test)