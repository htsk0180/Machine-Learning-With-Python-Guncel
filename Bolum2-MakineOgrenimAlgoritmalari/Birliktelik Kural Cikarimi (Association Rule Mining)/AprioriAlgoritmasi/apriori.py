# Apriori

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

veriler = pd.read_csv('sepet.csv', header = None) # header = None kolon başlığını ilk satır olarak algılamasın diye eklediğimiz parametredir.

# github üzerinden aldığımız algoritmaya parametre olarak vereceğimiz listoflist için veriler.csv'yi (eg. [['A', 'B'], ['B', 'C']]) şekline getiriyoruz.
t = []
for i in range (0,7501):
    t.append([str(veriler.values[i,j]) for j in range (0,20)])

from apyori import apriori
kurallar = apriori(t,min_support=0.01, min_confidence=0.2, min_lift = 3, min_length=2)
kurallar = list(kurallar)
pdKurallar = pd.DataFrame(kurallar)
