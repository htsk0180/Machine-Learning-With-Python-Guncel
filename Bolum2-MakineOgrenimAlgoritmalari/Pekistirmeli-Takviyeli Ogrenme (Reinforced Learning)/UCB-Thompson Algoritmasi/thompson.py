# -*- coding: utf-8 -*-
"""
Created on Thu Dec 30 12:16:41 2021

@author: 104863
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math
import random

veriler = pd.read_csv('Ads_CTR_Optimisation.csv')

# Thompson Algoritması
N = 10000 # işlem sayısı 
d = 10 # ilan sayısı 
toplam = 0 # toplam ödül.
secilenler = []
birler = [0] * d
sıfırlar = [0] * d

for n in range(1,N):
    ad = 0 # seçilen ilan
    max_th = 0
    # max ucb ye sahip ilanı bulacak olan döngü.
    for i in range(0,d):
        rasbeta = random.betavariate(birler[i]+1, sıfırlar[i] + 1)
        if rasbeta > max_th:
            max_th = rasbeta
            ad = i    
    secilenler.append(ad)
    odul = veriler.values[n,ad]
    if odul ==1:
        birler[ad] = birler[ad] + 1
    else:
        sıfırlar[ad] = sıfırlar[ad] +1
    toplam = toplam + odul
       
print(toplam)
plt.hist(secilenler)
plt.show()



