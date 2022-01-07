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


#ÖĞRENME
'''
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(x_train,y_train)
'''

#T AHMİN
#y_pred = regressor.predict(x_test)

# BOY TAHMİNİ

boy = s2.iloc[:,3:4].values
boyDf = pd.DataFrame(boy)
print(boyDf)

boyHaricDigerleri = s2.drop(["boy"],axis=1)
print(boyHaricDigerleri)

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(boyHaricDigerleri,boyDf,test_size=0.33,random_state=0)
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(x_train,y_train)

y_pred = regressor.predict(x_test)


# MODELİN BAŞARISINI ÖLÇMEK İÇİN
'''
import statsmodels.api as sm
X = np.append(arr= np.ones((22,1)).astype(int),values = boyHaricDigerleri, axis=1) # en başa birlerden oluşan bir dizi ekledik.
X_l = boyHaricDigerleri.iloc[:,[0,1,2,3,4,5]].values # p values hesaplaması yapabilmek için tablodaki kolonları aldık
X_l = np.array(X_l,dtype=float) # # p values hesaplaması yapabilmek için tablodaki kolonları aldık
model = sm.OLS(boy,X_l).fit() # boy dizisi ile diğer kalan dizileri statsmodels e verdik. bağımlı değişken boy ve bağımsız değişkenler olan diğer X_l dizisi
print(model.summary()) # 4. eleman 0.717 cıktı. algoritmaya göre en yüksek p değerine sahip olanı elememiz gerekecek.
'''

# 4 ü eliyoruz.
import statsmodels.api as sm
X = np.append(arr= np.ones((22,1)).astype(int),values = boyHaricDigerleri, axis=1) 
X_l = boyHaricDigerleri.iloc[:,[0,1,2,3,5]].values
X_l = np.array(X_l,dtype=float)

model = sm.OLS(boy,X_l).fit()
print(model.summary())
