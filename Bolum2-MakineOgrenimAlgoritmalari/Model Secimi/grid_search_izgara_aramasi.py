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
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state = 0)

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

##############################################################################
''' Model Seçimi ve Parametre İyileştirmesi:
    Grid Search ( Izgara Araması) Yöntemi.
    SVM algoritması ele alınırsa parametleri olan c ve kernel değerlerini optimize etmeye çalışacağız.
'''
from sklearn.model_selection import GridSearchCV
parametre = [{'C' : [1,2,3,4,5], 'kernel' : ['linear']},
             {'C' : [1,10,100,1000], 'kernel' : ['rbf'], 'gamma' : [1,0.5,0.1,0.01,0.001]}]

'''
GridSearchCV parametreleri:
estimator: hangi algoritmayı optimize etmek istiyoruz. Örneğimiz için sınıflandırma algoritması olan: SVC yani classifier.
param_grid: parametreler.
scoring: neye göre skorlanacak? Örneğin accuracy.
cv: kaç fold olacak.
n_jobs: aynı anda çalışacak iş. paralelleştirme. (şu anda yapmıyoruz.)
'''
gs = GridSearchCV(estimator=classifier, param_grid = parametre, scoring='accuracy', cv=10, n_jobs= -1)
grid_search = gs.fit(X_train, y_train)
eniyiskor = grid_search.best_score_
eniyiparam = grid_search.best_params_
print(eniyiskor)
print(eniyiparam)




