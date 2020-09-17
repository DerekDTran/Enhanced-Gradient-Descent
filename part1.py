# -*- coding: utf-8 -*-
"""
Created on Sun Sep 13 19:49:59 2020

@author: derek
"""
#Libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

#read csv file
Heart_DB = pd.read_csv("https://raw.githubusercontent.com/DerekDTran/Project1/master/heart_failure_clinical_records_dataset.csv")
print(Heart_DB.keys())
Heart_DB.dtypes

#preprocess the data
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

#split datasets into training and test datasets
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2, random_state=5)

#obtain MSE
def getLoss(X, Y, thetas):
    n = float(len(X))
    
    y_hat = X.dot(thetas)
    error_diff = np.subtract(y_hat, Y)
    
    SSE = np.sum(np.square(error_diff))
    MSE = SSE / (n)
    
    return MSE

#adaptive gradient descent
def AdaptGrad(X, Y, lr, itr):
    Sum_Gradient = 0
    error = []
    thetas = [np.random.normal(0, .001, (11,1))]
    
    for iteration in range(itr):
        for theta_itr in thetas:
            y_hat = X.dot(theta_itr)
            error_diff = np.subtract(y_hat, Y)
            grad = error_diff.T.dot(X)
            
            Sum_Gradient += (grad.T ** 2)
            theta_itr[:] -=  1/len(X) * grad.T * ((lr / np.sqrt(Sum_Gradient + 1e-6)))
            
            error.append(getLoss(X, Y, theta_itr))
    
    plt.figure()
    plt.plot(error)
    plt.xlabel('Num of Itr')
    plt.ylabel('Cost')
    
    return itr, theta_itr, error

#create header for file
Log = "number of iterations\tlr\tweights\tmse\n"
model = []

#process data through adaptive gradient
lrv = [.1, .01, .001, .0001]
for lr in lrv:
    itr, thetas_itr, mse = AdaptGrad(X_train, Y_train, lr, 300)
    object = {"number of iterations":itr,"lr":lr,"weights":thetas_itr,"mse":mse}
    model.append(object)
    
    Log = Log + str(itr) +"\t\t\t" + str(lr) +"\t" + str(thetas_itr) + "\t" + str(mse) + "\n\n"

#create log file
Log_File = open("log.txt", "w")
Log_File.write(Log)
Log_File.close()

#obtain mse and respective weights
itr_weight = []
itr_mse = []
optimal = []
for i in range(len(model)):
    itr_weight = model[i]['weights']
    itr_mse = model[i]['mse']
    
    optimal_itr = {"weights":itr_weight, "mse":float(itr_mse[99])}
    optimal.append(optimal_itr)

#find optimal weights related to lowest mse
lowest_mse = optimal[0]['mse']
for i in range(len(optimal)):
    if(lowest_mse > optimal[i]['mse']):
        lowest_mse = optimal[i]['mse']

optimal_weights = optimal[0]['weights']
for i in range(len(optimal)):
    if(lowest_mse == optimal[i]['mse']):
        optimal_weights = optimal[i]['weights']
print(optimal_weights)

#predict with the model
predict = []
X_list = X_test.values.tolist()

for i in range(len(X_list)):
    prediction = float(X_list[i][0] * optimal_weights[0] + X_list[i][1] * optimal_weights[1] + X_list[i][2] * optimal_weights[2] + X_list[i][3] * optimal_weights[3] + X_list[i][4] * optimal_weights[4] + X_list[i][5] * optimal_weights[5] + X_list[i][6] * optimal_weights[6] + X_list[i][7] * optimal_weights[7] + X_list[i][8] * optimal_weights[8] + X_list[i][9] * optimal_weights[9] + X_list[i][10] * optimal_weights[10])
    predict.append(prediction)
predict_list = [round(i) for i in predict]

print(predict_list)

#find error rate
correct = 0
Y_list = Y_test.values

for i in range(len(predict_list)):
    if(predict_list[i] == Y_list[i][0]):
        correct += 1
        
error = 1 - (correct/len(Y_list))
print(error)
print(lowest_mse)
