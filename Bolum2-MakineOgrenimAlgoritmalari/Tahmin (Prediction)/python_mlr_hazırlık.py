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

#eksik veriler.
imputer = SimpleImputer(missing_values=np.NAN,strategy="mean")
eksikveriler = pd.read_csv("veriler.csv")
imputer = SimpleImputer(missing_values=np.NAN,strategy="mean")
Yas = eksikveriler.iloc[:,1:4].values
imputer = imputer.fit(Yas[:,1:4]) 
Yas[:,1:4] = imputer.transform(Yas[:,1:4])

#ülke encoding
ulkeler = veriler.iloc[:, 0:1].values 
print(ulkeler)
from sklearn import preprocessing
le = preprocessing.LabelEncoder() # kolon şeklinde almak için kullandık.
ulkeler[:,0] = le.fit_transform(veriler.iloc[:,0]) # ilk ülkeye  sonra 2, 3 şeklinde sayısal verilere çevirdik.
ohe = preprocessing.OneHotEncoder() # 3 kolon haline getirdik. 
ulkeler = ohe.fit_transform(ulkeler).toarray()  # nupy dizisi şeklinde aldık.

#cinsiyette encoding
c = veriler.iloc[:,-1:].values
from sklearn import preprocessing
le = preprocessing.LabelEncoder()
c[:,-1] = le.fit_transform(veriler.iloc[:,-1])
ohe = preprocessing.OneHotEncoder()
c = ohe.fit_transform(c).toarray()
cinsiyetDf = pd.DataFrame(c)
cinsiyetDf = cinsiyetDf.drop([1], axis=1)

#dataframe 
sonucVeri = pd.DataFrame(data = ulkeler, index=range(22), columns=["fr","tr","us"])
print(sonucVeri)
sonucYas =  pd.DataFrame(data = Yas, index=range(22), columns=["boy","kilo","yas"])
print(sonucYas)
s = pd.concat([sonucVeri,sonucYas],axis=1)
print(s)
s2=pd.concat([s,cinsiyetDf],axis=1)
print(s2)

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(s,cinsiyetDf,test_size=0.33,random_state=0)



