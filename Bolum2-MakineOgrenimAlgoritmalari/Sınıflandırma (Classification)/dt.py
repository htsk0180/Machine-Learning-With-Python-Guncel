# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
#1- kütüphanaler:
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.impute import SimpleImputer

veriler = pd.read_csv("veriler.csv")
print(veriler)

x = veriler.iloc[:,1:4].values # bağımsız değişkenler.
y = veriler.iloc[:,4:].values # bağımlı değişkenler.

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.33,random_state=0)

from sklearn.preprocessing import StandardScaler

sc = StandardScaler()

# dikkat !! fit eğit demektir. sadece x_train'i fit edip transform ettik.
X_train = sc.fit_transform(x_train)
X_test = sc.transform(x_test)


from sklearn.linear_model import LogisticRegression
logr = LogisticRegression(random_state=0)
logr.fit(X_train,y_train)

y_pred = logr.predict(X_test)
print(y_pred)

# karmaşıklık matrisini hesaplama
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
print(cm) # çıkan sonuçta matrisin köşegeni bize doğru sınıflandıralan verinin sayısını verir.

# K-NN ALGOLİTMASI:
from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors=1, metric='minkowski') # kaç komşuya bakılacak ve aradaki ölçüm hangi algoritmaya göre yapılacak (minkowski)
knn.fit(X_train,y_train)

y_pred_knn = knn.predict(X_test)

from sklearn.metrics import confusion_matrix
cm_knn = confusion_matrix(y_test, y_pred_knn)
print(cm_knn)

# SVM ALGORİTMASI:
from sklearn.svm import SVC
svc = SVC(kernel='rbf')
svc.fit(X_train,y_train)

y_pred_svc = svc.predict(X_test)

from sklearn.metrics import confusion_matrix
cm_svc = confusion_matrix(y_test, y_pred_svc)
print(cm_svc)

# NB ALGORİTMASI:
# Gaussian Naive Bayes
from sklearn.naive_bayes import GaussianNB
gnb = GaussianNB()
gnb.fit(X_train, y_train)
y_pred_gnb = gnb.predict(X_test)
from sklearn.metrics import confusion_matrix
cm_gnb = confusion_matrix(y_test, y_pred_gnb)
print(cm_gnb)

# Bernoulli Naive Bayes
from sklearn.naive_bayes import BernoulliNB
bnb = BernoulliNB()
bnb.fit(X_train, y_train)
y_pred_bnb = bnb.predict(X_test)
from sklearn.metrics import confusion_matrix
cm_bnb = confusion_matrix(y_test, y_pred_bnb)
print(cm_bnb)

# DT ALGORİTMASI:
from sklearn.tree import DecisionTreeClassifier
dtc = DecisionTreeClassifier(criterion='entropy',random_state=0)
dtc.fit(X_train,y_train)
y_pred_dtc = dtc.predict(X_test)
from sklearn.metrics import confusion_matrix
cm_dtc = confusion_matrix(y_test, y_pred_dtc)
print(cm_dtc)




