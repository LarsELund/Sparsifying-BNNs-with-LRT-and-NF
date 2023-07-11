#!/usr/bin/env python3
# -*- coding: utf-8 -*-


### https://www.kaggle.com/code/tharunnayak14/knn-algorithm-0-842-accuracy#Create-2-sep-data-frames-one-with-numerical-data,-other-with-categorical-data


import pandas as pd
import numpy as np
filename = 'adult-all.csv'
# load the csv file as a data frame
data = pd.read_csv(filename,skiprows = 1)
data.columns =['age','workclass','fnlwgt','education','education-num',
               'marital-status','occupation','relationship','race','sex',
               'captial-gain','capital-loss','hours-per-week','native-country','income']

attrib, counts = np.unique(data['workclass'], return_counts = True)
most_freq_attrib = attrib[np.argmax(counts, axis = 0)]
data['workclass'][data['workclass'] == '?'] = most_freq_attrib 

attrib, counts = np.unique(data['occupation'], return_counts = True)
most_freq_attrib = attrib[np.argmax(counts, axis = 0)]
data['occupation'][data['occupation'] == '?'] = most_freq_attrib 

attrib, counts = np.unique(data['native-country'], return_counts = True)
most_freq_attrib = attrib[np.argmax(counts, axis = 0)]
data['native-country'][data['native-country'] == '?'] = most_freq_attrib 
data.income=data.income.replace(['<=50K', '>50K'],[0,1])
y = data.income
data.drop(columns=['income'],inplace = True)

df_cat = data.select_dtypes(include='object')
df_nums = data.select_dtypes(exclude='object')

df_cat = pd.get_dummies(df_cat)

df = pd.concat([df_nums,df_cat,y], axis=1)

df.to_csv('income_data.csv', index = False)