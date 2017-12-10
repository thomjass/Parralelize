# -*- coding: utf-8 -*-
"""
Created on Sun Dec 10 14:13:25 2017

@author: tjass
"""

import numpy as np
import random

#We assume that all the array are column array (x,1)
with open("C:\\Users\\tjass\\Documents\\Parallelize\\spam.txt") as file:
    X_temp = [];
    Y_temp = [];
    for line in file:
        # The rstrip method gets rid of the "\n" at the end of each line
        line_split = line.rstrip().split(" ")
        X_temp.append(line_split[:len(line_split)-2]);
        Y_temp.append([line_split[-1]])

        
# Find the min and max values for each column
def dataset_minmax(dataset):
    minmax = list();
    for i in range(len(dataset[0])):
        col_values = [float(row[i]) for row in dataset];
        value_min = min(col_values);
        value_max = max(col_values);
        minmax.append([value_min, value_max]);
    return minmax;

def normalize_dataset(dataset, minmax):
    for row in dataset:
        for i in range(len(row)):
            row[i] = (float(row[i]) - minmax[i][0]) / (minmax[i][1] - minmax[i][0]);



minmax = dataset_minmax(X_temp)
normalize_dataset(X_temp,minmax)

X_tp = np.asarray(X_temp)
Y_tp = np.asarray(Y_temp)

## Split the data set into two parts: train and test

split_index = int(0.75*len(X_tp));
train_X = X_tp[:split_index];
train_Y = Y_tp[:split_index];
test_X = X_tp[split_index:];
test_Y = Y_tp[split_index:];



def sigma(z):
    if np.any(1/(1+np.exp(-z))>=0.5):
        return 1.
    else:
        return 0.       
        
def cost_function(y_estim,y,W,lambda_reg):
        summ=0.
        for m in range(0,len(y)):
            if y_estim[m][0]==0:
                summ = summ +(1.-float(y[m][0]))
            else:
                summ = summ + float(y[m][0])*np.log(y_estim[m][0]) + (1.-float(y[m][0]))*(1.-np.log(y_estim[m][0]))
        
        res = (1./len(y))*summ + (lambda_reg/(2.*len(y)))*np.sum(np.power(W,2))
        return res
#X est du type numpy array
def train(X,Y,iterations, learning_rate, lambda_reg):
    #Initialisation
    W = np.random.rand(56,1)
    b= random.random()
    #Process
    for it in range(iterations):
        dW = np.zeros((57,1))
        y_estim = np.zeros((len(X),1))
        summ = 0.
        summ_b = 0.
        for k in range(0,len(W)):
            for m in range(0,len(X)):
                y_inter = np.dot(np.transpose(W),X[m])
                y_estim[m]=sigma(y_inter[0]+b)
                summ = summ + (y_estim[m] - float(Y[m][0]))*X[m,k]
                summ_b = summ_b + (y_estim - float(Y[m][0]))
            dW[k]=(1./len(X))*summ+(lambda_reg/len(X))*W[k]
            W[k] = W[k] - learning_rate * dW[k]
        db=(1./len(X))*summ_b
        b= b-learning_rate*db
        print(cost_function(y_estim,Y,W,lambda_reg))
    return (W,b)

def predict(W,b,X):
    y_predict= np.zeros((len(X),1))
    for m in range(0,len(X)):
        y_inter = np.dot(np.transpose(W),X[m])
        y_predict[m]=sigma(y_inter[0]+b)
    return y_predict


def accuracy_metric(actual, predicted):
    correct = 0;
    for i in range(len(actual)):
        if float(actual[i][0]) == predicted[i][0]:
            correct += 1;
    return correct / float(len(actual)) * 100.0;


(W,b) = train(train_X,train_Y,5,0.1,0.1)
y_predict = predict(W,b,test_X)
print("Accuracy Metric")
print(accuracy_metric(test_Y,y_predict))
