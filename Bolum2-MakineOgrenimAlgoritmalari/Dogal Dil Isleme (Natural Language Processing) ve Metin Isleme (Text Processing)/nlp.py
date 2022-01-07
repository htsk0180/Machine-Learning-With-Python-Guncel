# -*- coding: utf-8 -*-
"""
Created on Fri Dec 31 20:57:50 2021

@author: 104863
"""

import numpy as np
import pandas as pd
import re   

yorumlar = pd.read_csv('Restaurant_Reviews.csv',on_bad_lines='skip')
yorumlar.fillna(0, inplace=True)

# 1-) VERİ ÖN İŞLEME: 
    
# nltk kullanımı:
import nltk
# ingilizce için stop word kelimelerin indirdik. sonrasında PorterStemmer ile birlikte kullanım yaparak(highly deki ly sileceğiz.)
# did not gibi stopwords kelimeleri de almayacağız.
from nltk.stem.porter import PorterStemmer # kelimeyi köklerine ayırma işlemi.
ps = PorterStemmer()
nltk.download('stopwords')
from nltk.corpus import stopwords # corpus üzerinden ekledik.
derleme = []
for i in range(716):
    # imla işaratleri ve alfanümerik verilerin filitrelenmesi, sparce matrix:
    yorum = re.sub('[^a-zA-Z]',' ',yorumlar['Review'][i])
    # büyük/küçük harf problemi:
    yorum = yorum.lower()
    yorum = yorum.split()
    # stopwords kelime gruplarından kontrol ederek yeni liste oluşturuyoruz.
    yorum = [ps.stem(kelime) for kelime in yorum if not kelime in set(stopwords.words('english'))]
    yorum = ' '.join(yorum) # parçalı listeyi tek bir str ye dönüştürüyoruz.
    derleme.append(yorum)
derlemeyorum = pd.DataFrame(derleme)

# 2-) ÖZNİTELİK ÇIKARIMI:

# Kelime Vektörü Sayaç kullanımı (CountVectorizer)
from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(max_features=2000)# en fazla kullanılan 1000 kelimeyi al. 
#bağımsız değişken:
X = cv.fit_transform(derleme).toarray() # x ekseni kelime, y ekseni yorum : sparce matrisi elde ettik.
#bağımlı değişken:
Y = yorumlar.iloc[:,1].values

# 3-) MAKİNE ÖĞRENİMİ:
    
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(X,Y,test_size=0.33,random_state=0)

# Bernoulli Naive Bayes
from sklearn.naive_bayes import BernoulliNB
bnb = BernoulliNB()
bnb.fit(x_train, y_train)
y_pred_bnb = bnb.predict(x_test)
from sklearn.metrics import confusion_matrix
cm_bnb = confusion_matrix(y_test, y_pred_bnb)
print(cm_bnb) # %76 accuracy. 

#ysa
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
X_train = scaler.fit_transform(x_train)

X_test = scaler.fit_transform(x_test)

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
model = Sequential()

model.add(Dense(12,activation="relu"))
model.add(Dense(12,activation="relu"))
model.add(Dense(12,activation="relu"))
model.add(Dense(12,activation="relu"))

model.add(Dense(1))
model.compile(optimizer="adam" , loss="mse")

model.fit(x = X_train, y = y_train ,validation_data=(X_test,y_test), epochs=300)
kayipVerisi = pd.DataFrame(model.history.history)
kayipVerisi.head()
kayipVerisi.plot()
tahminDizisi = model.predict(X_test)


