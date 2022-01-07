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

from sklearn.cluster import KMeans

kmeans = KMeans(n_clusters=4, init='k-means++')
kmeans.fit(X)

y_pred = kmeans.predict([['78852','5155']])
print(f'dahil olunan cluster: {y_pred}')
print(kmeans.cluster_centers_)

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

    
    




