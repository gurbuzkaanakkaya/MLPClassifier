import pandas as pd
from pandas import DataFrame
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import GradientBoostingClassifier
from sklearn import metrics
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import precision_score, recall_score
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D,BatchNormalization
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.utils import to_categorical
import seaborn as sns
import glob
import pickle
import os
import uuid
from mss import mss
from collections import deque


dataset = pd.read_csv("C:/Users/.../data.csv")
labelset = pd.read_csv("C:/Users/.../labels.csv")

datasetfex = pd.read_csv('data.csv',delimiter=",")
labelsetfex = pd.read_csv('labels.csv',names=["samples",""],delimiter=",")

X_f = datasetfex.iloc[:, 1:1837].values
y = labelsetfex.iloc[1:, 1].values
for i in range(0,len(y)):
    if y[i] == "colon cancer":
        y[i] = 0
    if y[i] == "lung cancer":
        y[i] = 1
    if y[i] == "breast cancer":
        y[i] = 2
    if y[i] == "prosrtate cancer":
        y[i] = 3


        
Y_Norm = y.copy()
Y_Norm = pd.DataFrame(Y_Norm)
Y_Norm = Y_Norm.apply(pd.to_numeric) 
X_Norm = X_f.copy()
X_Norm = pd.DataFrame(X_Norm)
X_Norm = X_Norm.apply(pd.to_numeric)

for column in X_Norm.columns:
    X_Norm[column] = (X_Norm[column] - X_Norm[column].min()) / (X_Norm[column].max() - X_Norm[column].min())
X = X_Norm.values
y = Y_Norm.values
y = np.ravel(y)
results = []

dataset.columns.values[0] = 'Sample'

f_space = dataset.drop('Sample', axis = 1)
f_class = labelset['disease_type']

x_train, x_test, y_train, y_test = train_test_split(f_space, f_class, test_size = 0.5, random_state = 42)
y_train = y_train.values.ravel() 
y_test = y_test.values.ravel()

print("MODEL SHAPES")
print('Training Features Shape:', x_train.shape)
print('Training Labels Shape:', y_train.shape)
print('Testing Features Shape:', x_test.shape)
print('Testing Labels Shape:', y_test.shape)

scaler = StandardScaler()
scaler.fit(x_train)

x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)

MLPClassifier_1 = MLPClassifier(activation = 'relu',alpha=1e-5, hidden_layer_sizes = (100, 50, 25), max_iter = 100, random_state = None, solver = 'lbfgs')
MLPClassifier_1.fit(x_train, y_train)

predictions = MLPClassifier_1.predict(x_test)
precision = precision_score(y_test, predictions, average = None)
print("-------------------------------------------------")
print("MLPClassifier_1 Değişkeninin Performans Sonuçları")
print("PRECISION:", precision)

recall = recall_score(y_test, predictions, average = None)

print("RECALL:", recall)

F2_measure = 5 * precision * recall / (4 * precision + recall)

print("F2 MEASURE:", F2_measure)
results.append([precision,recall,F2_measure])

MLPClassifier_2 = MLPClassifier(activation = 'logistic',alpha=1e-5, hidden_layer_sizes = (100, 50, 25), max_iter = 500, random_state = 1, solver = 'adam')
MLPClassifier_2.fit(x_train, y_train)

predictions = MLPClassifier_2.predict(x_test)
precision = precision_score(y_test, predictions, average = None)
print("-------------------------------------------------")
print("MLPClassifier_2 Değişkeninin Performans Sonuçları")
print("PRECISION:", precision)

recall = recall_score(y_test, predictions, average = None)

print("RECALL:", recall)

F2_measure = 5 * precision * recall / (4 * precision + recall)

print("F2 MEASURE:", F2_measure)
results.append([precision,recall,F2_measure])

MLPClassifier_3 = MLPClassifier(activation = 'identity',alpha=1e-5, hidden_layer_sizes = (100, 50, 25), max_iter = 900, random_state = 2, solver = 'sgd')
MLPClassifier_3.fit(x_train, y_train)

predictions = MLPClassifier_3.predict(x_test)
precision = precision_score(y_test, predictions, average = None)
print("-------------------------------------------------")
print("MLPClassifier_3 Değişkeninin Performans Sonuçları")
print("PRECISION:", precision)

recall = recall_score(y_test, predictions, average = None)

print("RECALL:", recall)

F2_measure = 5 * precision * recall / (4 * precision + recall)

print("F2 MEASURE:", F2_measure)
results.append([precision,recall,F2_measure])

MLPClassifier_4 = MLPClassifier(activation = 'tanh',alpha=1e-5, hidden_layer_sizes = (100, 50, 25), max_iter = 1200, random_state = 3, solver = 'adam')
MLPClassifier_4.fit(x_train, y_train)

predictions = MLPClassifier_4.predict(x_test)
precision = precision_score(y_test, predictions, average = None)
print("-------------------------------------------------")
print("MLPClassifier_4 Değişkeninin Performans Sonuçları")
print("PRECISION:", precision)

recall = recall_score(y_test, predictions, average = None)

print("RECALL:", recall)

F2_measure = 5 * precision * recall / (4 * precision + recall)

print("F2 MEASURE:", F2_measure)
results.append([precision,recall,F2_measure])
print("-------------------------------------------------")

df = DataFrame(results,columns=["Precision","Recall","F2 Measure"])
df.to_excel ('algorithmMLPresults.xlsx', index = False, header=True)



