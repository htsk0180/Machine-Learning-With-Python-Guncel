# -*- coding: utf-8 -*-
"""
Created on Thu Dec 30 12:16:41 2021

@author: 104863
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math

veriler = pd.read_csv('Ads_CTR_Optimisation.csv')

# Random Selection (Rastgele Seçim)
'''
import random
N = 10000
d = 10 
toplam = 0
secilenler = []
for n in range(0,N):
    ad = random.randrange(d)
    secilenler.append(ad)
    odul = veriler.values[n,ad] # verilerdeki n. satır = 1 ise odul 1
    toplam = toplam + odul
plt.hist(secilenler)
plt.show()
'''

# UCB Algoritması
N = 10000
d = 10 
oduller = [0] * d  # 10 elamanı da 0 olan liste yaptık.
tiklamar = [0] * d  # 10 elamanı da 0 olan liste yaptık.
toplam = 0 # toplam ödül.
secilenler = []

for n in range(0,N):
    ad = 0 # seçilen ilan
    max_ucb = 0
    # max ucb ye sahip ilanı bulacak olan döngü.
    for i in range(0,d):
        if (tiklamar[i] > 0):
            ortalama = oduller[i]/tiklamar[i]
            delta = math.sqrt(3/2 * math.log(n)/tiklamar[i])
            ucb = ortalama + delta
        else:
            ucb = N*10
        if max_ucb < ucb:
            max_ucb = ucb
            ad = i
    secilenler.append(ad)
    tiklamar[ad] = tiklamar[ad] + 1
    odul = veriler.values[n,ad]
    oduller[ad] = oduller[ad] + odul
    toplam = toplam + odul
       
print(toplam)
plt.hist(secilenler)
plt.show()



