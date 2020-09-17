# -*- coding: utf-8 -*-
"""
Created on Sun Sep 13 21:09:45 2020

@author: derek
"""

#libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from sklearn.linear_model import SGDRegressor
from sklearn import metrics

#read csv file
Heart_DB = pd.read_csv("https://raw.githubusercontent.com/DerekDTran/Project1/master/heart_failure_clinical_records_dataset.csv")
print(Heart_DB.keys())
Heart_DB.dtypes

#chekc for missing and null values
Heart_DB.isna().sum()
Heart_DB.isnull().sum()

#plot for any dependencies regarding response variable
sns.set(rc={'figure.figsize':(11.7,8.27)})
sns.distplot(Heart_DB['DEATH_EVENT'], bins=30)
plt.show()

#plot heat map
correlation_matrix = Heart_DB.corr().round(2)
sns.heatmap(data=correlation_matrix, annot=True)

#remove redundant variables
Heart_DB.drop(['time'], axis=1)

#obtain predictors and response variables
X = pd.DataFrame(np.c_[Heart_DB['age'], Heart_DB['anaemia'], Heart_DB['creatinine_phosphokinase'], Heart_DB['diabetes'], Heart_DB['ejection_fraction'], Heart_DB['high_blood_pressure'], Heart_DB['platelets'], Heart_DB['serum_creatinine'], Heart_DB['serum_sodium'], Heart_DB['sex'], Heart_DB['smoking']], columns = ['age', 'anaemia', 'creatinine_phosphokinase', 'diabetes', 'ejection_fraction', 'high_blood_pressure', 'platelets', 'serum_creatinine', 'serum_sodium', 'sex', 'smoking'])
Y = Heart_DB[['DEATH_EVENT']]
X = X.values.tolist()
Y = Y.values.flatten()

#split datasets into training and test datasets
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2, random_state=5)

#create and fit SGDRegressor into model with data
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
regression = make_pipeline(StandardScaler(), SGDRegressor(max_iter=1000, tol=1e-3))
regression.fit(X, Y)

#model predictions on test datat
predict = regression.predict(X_test)
predict_list = [round(i) for i in predict]
print(predict_list)

#find error rate
correct = 0

for i in range(len(predict_list)):
    if(predict_list[i] == Y_test[i]):
        correct += 1
        
error = 1- (correct/len(Y_test))
print(error)
print(metrics.mean_squared_error(Y_test, predict))
