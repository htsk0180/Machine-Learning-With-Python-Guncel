# -*- coding: utf-8 -*-
"""
Created on Wed Jan  5 14:51:13 2022

@author: 104863
"""

# Kutuphaneler
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Veri kümesi
dataset = pd.read_csv('Social_Network_Ads.csv')
X = dataset.iloc[:, [2, 3]].values
y = dataset.iloc[:, 4].values

# Eğitim ve test kümelerinin bölünmesi
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.68, random_state = 0)

# Ölçekleme
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# SVM
from sklearn.svm import SVC
classifier = SVC(kernel = 'rbf', random_state = 0)
classifier.fit(X_train, y_train)

# Tahminler
y_pred = classifier.predict(X_test)

#  Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)

print(cm)


# K-katlamali capraz dogrulama 
from sklearn.model_selection import cross_val_score
''' 
cross_val_score parametreleri:
1.parametre: estimator : classifier (svm için. farklı bir algoritma kullanırsa o yazılır.)
2.parametre: X
3.parametre: Y
4.parametre: cv : kaç katlamalı
'''
basari = cross_val_score(estimator = classifier, X=X_train, y=y_train , cv = 4)
print(f' algoritmanın başarısı: {basari.mean()}') 
print(f' algoritmanın başarısının standart sapması: {basari.std()}')









