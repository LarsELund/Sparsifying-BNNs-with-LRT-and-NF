from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold
import pandas as pd
import os
import numpy as np

data = pd.read_csv('income_data.csv', sep = ",") #get a round number


full_data = np.array(data)
p = full_data.shape[1] - 1


X = full_data[:,0:p]
y = full_data[:,-1]
skf = StratifiedKFold(n_splits=10,shuffle = True,random_state = 1)

boost_scores = np.zeros(10)
forest_scores = np.zeros(10)
depth = 10

for i, (train_index, test_index) in enumerate(skf.split(X, y)):
        
        train_dat,y_train = X[train_index],y[train_index]
        test_dat,y_test = X[test_index],y[test_index]
        m1,s1 = train_dat[:,0].mean(),train_dat[:,0].std()
        m2,s2 = train_dat[:,1].mean(),train_dat[:,1].std()
        m3,s3 = train_dat[:,2].mean(),train_dat[:,2].std()
        m4,s4 = train_dat[:,3].mean(),train_dat[:,3].std()
        m5,s5 = train_dat[:,4].mean(),train_dat[:,4].std()
        m6,s6 = train_dat[:,5].mean(),train_dat[:,5].std()


        train_dat[:,0] = (train_dat[:,0] - m1) / s1
        train_dat[:,1] = (train_dat[:,1] - m2) / s2
        train_dat[:,2] = (train_dat[:,2] - m3) / s3
        train_dat[:,3] = (train_dat[:,3] - m4) / s4
        train_dat[:,4] = (train_dat[:,4] - m5) / s5
        train_dat[:,5] = (train_dat[:,5] - m6) / s6

        test_dat[:,0] = (test_dat[:,0] - m1) / s1
        test_dat[:,1] = (test_dat[:,1] - m2) / s2
        test_dat[:,2] = (test_dat[:,2] - m3) / s3
        test_dat[:,3] = (test_dat[:,3] - m4) / s4
        test_dat[:,4] = (test_dat[:,4] - m5) / s5
        test_dat[:,5] = (test_dat[:,5] - m6) / s6
        clf = GradientBoostingClassifier(n_estimators= 100, learning_rate=0.5,
         max_depth=depth, random_state=i).fit(train_dat, y_train)
      
        clf_f = RandomForestClassifier(n_estimators = 100,max_depth=depth, criterion='gini', random_state=i).fit(train_dat,y_train)
        boost_scores[i] = clf.score(test_dat, y_test)
        forest_scores[i] = clf_f.score(test_dat, y_test)
        print(boost_scores[i])
        
