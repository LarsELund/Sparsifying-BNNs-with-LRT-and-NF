import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold
import os



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


boost_scores = np.zeros(10)
forest_scores = np.zeros(10)

X = full_data[:,0:p]
y = full_data[:,-1]
skf = StratifiedKFold(n_splits=10,shuffle = True,random_state = 1)
depth = 1000

learning_rate = 0.5


for i, (train_index, test_index) in enumerate(skf.split(X, y)):
        
        X_train,y_train = X[train_index],y[train_index]
        X_test,y_test = X[test_index],y[test_index]
        
        mu,std = X_train.mean(axis = 0), X_train.std(axis = 0)
        X_tr = (X_train - mu) / std
        X_te = (X_test  - mu) / std
        
        
        clf = GradientBoostingClassifier(n_estimators=100,min_samples_split =5,min_samples_leaf = 5, max_depth=depth,learning_rate=learning_rate, random_state=i).fit(X_tr, y_train)
        clf_f = RandomForestClassifier(n_estimators = 100,max_depth=depth, random_state=i).fit(X_tr,y_train)
        boost_scores[i] = clf.score(X_te, y_test)
        forest_scores[i] = clf_f.score(X_te, y_test)
        print(boost_scores[i],forest_scores[i])

a = np.array(boost_scores.mean()).reshape(1,1) #to be able to save it as an array
b= np.array(forest_scores.mean()).reshape(1,1)
print(a,b)


current_dir = os.getcwd()
np.savetxt(current_dir +'/results/xgboost_avg_metrics.txt',a,delimiter = ',')
np.savetxt(current_dir +'/results/rf_avg_metrics.txt',b,delimiter = ',')