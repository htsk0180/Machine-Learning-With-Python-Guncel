# -*- coding: utf-8 -*-
"""
Created on Fri Dec 31 20:57:50 2021

@author: 104863
"""

# Kutuphaneler
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Veri kümesi
veriler = pd.read_csv('Wine.csv')
X = veriler.iloc[:, 0:13].values
y = veriler.iloc[:, 13].values

# Eğitim ve test kümelerinin bölünmesi
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

# Ölçekleme
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# PCA

from sklearn.decomposition import PCA
pca = PCA(n_components=2) # kaç kolona indirgemek istediğimiz: n_components
X_train2 = pca.fit_transform(X_train) # X_train'deki veriler ile kendini eğittikten sonra trans ederek yeni X_train'i elde edecektir.
X_test2 = pca.transform(X_test) # fit yapmamamızdaki sebep ise zaten train kısmında pca kendini eğitti.


from sklearn.linear_model import LogisticRegression

# PCA dönüşüm öncesi LR
classifier = LogisticRegression(random_state=0) # birazdan da yine logisticregression kullanacağımız için aynı yapının gitmesini istediğimizden dolayı random state verdik.
classifier.fit(X_train, y_train)

# PCA dönüşüm sonrasında LR
classifier2 = LogisticRegression(random_state=0)
classifier2.fit(X_train2, y_train)

# Tahminler 
y_pred = classifier.predict(X_test)
y_pred2 = classifier2.predict(X_test2)

# Karşılaştırma
from sklearn.metrics import confusion_matrix

# Actual / PCA olmadan çıkan sonuç
print('Gerçek / PCA-sız')
cm = confusion_matrix(y_test,y_pred)
print(cm)

# Actual / PCA sonrası çıkan sonuç
print("Gerçek / PCA ile")
cm2 = confusion_matrix(y_test,y_pred2)
print(cm2)

# PCA sonrası / PCA öncesi
print('PCA-sız / PCA-lı')
cm3 = confusion_matrix(y_pred,y_pred2)
print(cm3)

##############################################################################

# LDA

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
lda = LinearDiscriminantAnalysis(n_components = 2 )
X_train_lda = lda.fit_transform(X_train, y_train) # pca 'dan farklı olarak bir de y_train verdik. çünkü lda'nın çalışabilmesi için sınıfları öğrenmesi gerekir.
# lda sınıflar arasındaki farkları maximize ediyor.  
X_test_lda = lda.transform(X_test)

# LDA dönüşümünden sonra LR
classifier_lda = LogisticRegression(random_state=0)
classifier_lda.fit(X_train_lda, y_train)

# Tahmin
y_pred_lda = classifier_lda.predict(X_test_lda)

# Orjinal / LDA-lı
print('Orjinal / LDA-lı')
cm3_lda = confusion_matrix(y_pred,y_pred_lda)
print(cm3_lda)








