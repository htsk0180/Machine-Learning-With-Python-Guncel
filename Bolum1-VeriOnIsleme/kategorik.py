# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.impute import SimpleImputer

veriler = pd.read_csv("veriler.csv")
print(veriler)
imputer = SimpleImputer(missing_values=np.NAN,strategy="mean")

# yaş verilerindeki eksik satırların ortalama ile doldurulması.
eksikveriler = pd.read_csv("eksikveriler.csv")
imputer = SimpleImputer(missing_values=np.NAN,strategy="mean")
Yas = eksikveriler.iloc[:,1:4].values
imputer = imputer.fit(Yas[:,1:4]) # simpleImputer daki stratejimiz mean olduğunu için bu kolondaki ortalama değerleri öğrenecek.
Yas[:,1:4] = imputer.transform(Yas[:,1:4])

# ulkeleri çoğu makine öğrenmesi modellerinde kullanacağımız şekilde sayısal hale getireceğiz.
# kategorik veriden numeric veriye dönüşüm işlemini yapacağız.

ulkeler = veriler.iloc[:, 0:1].values 
print(ulkeler)

from sklearn import preprocessing
le = preprocessing.LabelEncoder() # kolon şeklinde almak için kullandık.

ulkeler[:,0] = le.fit_transform(veriler.iloc[:,0]) # ilk ülkeye  sonra 2, 3 şeklinde sayısal verilere çevirdik.
print(ulkeler)

ohe = preprocessing.OneHotEncoder() # 3 kolon haline getirdik. 
ulkeler = ohe.fit_transform(ulkeler).toarray()  # nupy dizisi şeklinde aldık.
print(ulkeler)



