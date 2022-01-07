# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
#1- kütüphanaler:
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

veriler = pd.read_csv("musteriler.csv")
print(veriler)

X = veriler.iloc[:,3:].values

# K-Means Kümeleme Algoritması:
from sklearn.cluster import KMeans

kmeans = KMeans(n_clusters=4, init='k-means++')
kmeans.fit(X)

# K için Optimum Değeri Bulmak İstiyoruz (WCSS)
'''
sonuclar = []
for i in range(1,11):
    kmeans = KMeans(n_clusters=i,init='k-means++',random_state=0)
    kmeans.fit(X)
    sonuclar.append(kmeans.inertia_)# inertia_ metodu wcss değerlerimiz.
print(sonuclar)
plt.plot(range(1,11), sonuclar)
'''

y_pred = kmeans.predict([['78852','5155']])
print(f'dahil olunan cluster: {y_pred}')
print(kmeans.cluster_centers_)

kmeans = KMeans(n_clusters=4, init='k-means++')
Y_tahminKM = kmeans.fit_predict(X)
plt.scatter(X[Y_tahminKM==0,0] , X[Y_tahminKM==0,1],s=100,c='yellow')
plt.scatter(X[Y_tahminKM==1,0] , X[Y_tahminKM==1,1],s=100,c='black')
plt.scatter(X[Y_tahminKM==2,0] , X[Y_tahminKM==2,1],s=100,c='orange')
plt.scatter(X[Y_tahminKM==3,0] , X[Y_tahminKM==3,1],s=100,c='red')
plt.show()

# Hiyerarşik Kümeleme Algoritması:
from sklearn.cluster import AgglomerativeClustering
ac = AgglomerativeClustering(n_clusters=3 , affinity='euclidean' , linkage = 'ward'  )
Y_tahmin = ac.fit_predict(X)
print(Y_tahmin)
plt.scatter(X[Y_tahmin==0,0] , X[Y_tahmin==0,1],s=100,c='red')
plt.scatter(X[Y_tahmin==1,0] , X[Y_tahmin==1,1],s=100,c='green')
plt.scatter(X[Y_tahmin==2,0] , X[Y_tahmin==2,1],s=100,c='blue')
plt.scatter(X[Y_tahminKM==3,0] , X[Y_tahminKM==3,1],s=100,c='yellow')
plt.show()

# SciPy Kütüphanesi ,dendrogram çizimi için..
import scipy.cluster.hierarchy as sch
dendrogram = sch.dendrogram(sch.linkage(X, method='ward'))
print(dendrogram)
plt.show()



    
    




