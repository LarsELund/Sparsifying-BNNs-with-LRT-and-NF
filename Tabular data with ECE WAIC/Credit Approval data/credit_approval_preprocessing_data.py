#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
from sklearn.preprocessing import OrdinalEncoder
np.random.seed(1)
# Load dataset
#df = pd.read_csv('../input/ucidata/crx.data',header=None)
header_names=['A','B','C','D','E','F','G','H','I','J','K','L','M','N','O','P']
df = pd.read_csv('crx.data',names=header_names)
df = df.replace('?',np.nan)

# https://medium.com/@amansangal9/predicting-credit-card-approvals-8409c5280f91

def fix_missing_mean(df,col):
    ''' This function takes a data frame as input 
    replaces the missing values of a particular column with it's mean value
    '''
    #replace missing values with mean 
    df[col] = pd.to_numeric(df[col], errors = 'coerce')
    df[col].fillna(df[col].mean(), inplace = True)    

def fix_missing_ffill(df, col):
    ''' This function takes a data frame as input 
    replaces the missing values of a particular column with the value from the previous row
    '''
    df[col] = df[col].fillna(method='ffill')  
    
fix_missing_ffill(df,'A')
fix_missing_ffill(df,'B')
fix_missing_ffill(df,'D')
fix_missing_ffill(df,'E')
fix_missing_ffill(df,'F')
fix_missing_ffill(df,'G')
fix_missing_mean(df,'N')


# Separate target from features
y = df['P']
y = y.replace('+',1)
y = y.replace('-',0)
features = df.drop(['P'], axis=1)
# Preview features
features.head()


# List of categorical columns
object_cols = ['A','B','D','E','F','G','I','J','L','M','N']

# ordinal-encode categorical columns
X = features.copy()
ordinal_encoder = OrdinalEncoder()
X[object_cols] = ordinal_encoder.fit_transform(features[object_cols])

dat = np.column_stack((X,y))
np.savetxt('dat.txt',dat,delimiter=',')

 #change it into array
np.random.shuffle(dat)
train = dat[0:500,:]
test = dat[500:,:]

#standardize the continous variables

m1,s1 = train[:,1].mean(),train[:,1].std()
m2,s2 = train[:,2].mean(),train[:,2].std()
m7,s7 = train[:,7].mean(),train[:,7].std()
m13,s13 = train[:,13].mean(),train[:,13].std()
m14,s14 = train[:,14].mean(),train[:,14].std()


train[:,1] = (train[:,1] - m1) / s1
train[:,2] = (train[:,2] - m2) / s2
train[:,7] = (train[:,7] - m7) / s7
train[:,13] = (train[:,13] - m13) / s13
train[:,14] = (train[:,14] - m14) / s14

test[:,1] = (test[:,1] - m1) / s1
test[:,2] = (test[:,2] - m2) / s2
test[:,7] = (test[:,7] - m7) / s7
test[:,13] = (test[:,13] - m13) / s13
test[:,14] = (test[:,14] - m14) / s14

np.savetxt('train.txt',train,delimiter=',')
np.savetxt('test.txt',test,delimiter=',')











