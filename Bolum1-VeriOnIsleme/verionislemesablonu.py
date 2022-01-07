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

#2- veri ön işleme:

#2.1- veri yükleme:
veriler = pd.read_csv("veriler.csv")
print(veriler)

#3- eksik veriler:
imputer = SimpleImputer(missing_values=np.NAN,strategy="mean")
# yaş verilerindeki eksik satırların ortalama ile doldurulması.
eksikveriler = pd.read_csv("eksikveriler.csv")
imputer = SimpleImputer(missing_values=np.NAN,strategy="mean")
Yas = eksikveriler.iloc[:,1:4].values
imputer = imputer.fit(Yas[:,1:4]) # simpleImputer daki stratejimiz mean olduğunu için bu kolondaki ortalama değerleri öğrenecek.
Yas[:,1:4] = imputer.transform(Yas[:,1:4])

#4- encoder: 
#Nominal,Ordinal(katagorik verilerden ) - > Numeric
# ulkeleri çoğu makine öğrenmesi modellerinde kullanacağımız şekilde sayısal hale getireceğiz.
# kategorik veriden numeric veriye dönüşüm işlemini yapacağız.
ulkeler = veriler.iloc[:, 0:1].values 
print(ulkeler)
from sklearn import preprocessing
le = preprocessing.LabelEncoder() # kolon şeklinde almak için kullandık.
ulkeler[:,0] = le.fit_transform(veriler.iloc[:,0]) # ilk ülkeye  sonra 2, 3 şeklinde sayısal verilere çevirdik.
ohe = preprocessing.OneHotEncoder() # 3 kolon haline getirdik. 
ulkeler = ohe.fit_transform(ulkeler).toarray()  # nupy dizisi şeklinde aldık.

#5- DataFrame Dönüşümü:     
#sonucu tek bir df de toplayacağız.
sonucVeri = pd.DataFrame(data = ulkeler, index=range(22), columns=["fr","tr","us"])
print(sonucVeri)
sonucYas =  pd.DataFrame(data = Yas, index=range(22), columns=["boy","kilo","yas"])
print(sonucYas)
cinsiyet = veriler.iloc[:, -1].values 
print(cinsiyet)
sonucCinsiyet = pd.DataFrame(data = cinsiyet, index=range(22), columns=["cinsiyet"])
print(sonucCinsiyet)

#6- DataFrame Birleştirme:
s = pd.concat([sonucVeri,sonucYas],axis=1)
print(s)
s2=pd.concat([s,sonucCinsiyet],axis=1)
print(s2)

#7- Test/Train olarak veri bölme işlemi:
# verilerimizi eğitim ve test verisi olarak böleceğiz.
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(s,sonucCinsiyet,test_size=0.33,random_state=0)

#8- Scaler İşlemi:
# verileri dönüştürme.
# verileri 0-1 arasında bir değere dönüştürmeye çalışıyoruz. scaler işlemi yapacağız.
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(x_train)
X_test = sc.fit_transform(x_test)




