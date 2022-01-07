# -*- coding: utf-8 -*-
"""
Created on Fri Dec 31 20:57:50 2021

@author: 104863
"""
# Kutuphaneler
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Veri yükleme
veriler = pd.read_csv('Churn_Modelling.csv')

# Veri on isleme
X= veriler.iloc[:,3:13].values
Y = veriler.iloc[:,13].values

# Encoder: Kategorik -> Numeric
from sklearn import preprocessing
le = preprocessing.LabelEncoder()
X[:,1] = le.fit_transform(X[:,1])
le2 = preprocessing.LabelEncoder()
X[:,2] = le2.fit_transform(X[:,2])

from sklearn.compose import ColumnTransformer # birden fazla kolonun aynı anda ayrı ayrı dönüştürülmesi.  
from sklearn.preprocessing import OneHotEncoder
ohe = ColumnTransformer([("ohe", OneHotEncoder(dtype=float),[1])],
                        remainder="passthrough"
                        )
X = ohe.fit_transform(X)
X = X[:,1:]

# Verilerin egitim ve test icin bolunmesi
from sklearn.model_selection import train_test_split
x_train, x_test,y_train,y_test = train_test_split(X,Y,test_size=0.33, random_state=0)

# XGBoost Kullanımı
from xgboost import XGBClassifier
classifier = XGBClassifier()
classifier.fit(x_train,y_train)

y_pred = classifier.predict(x_test)

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_pred,y_test)
print(cm)

# AdaBoost  Kullanımı
from sklearn.ensemble import AdaBoostClassifier
ada_classifier = AdaBoostClassifier(n_estimators=50, random_state=20)
ada_classifier.fit(x_train,y_train)

y_pred_ada = classifier.predict(x_test)

from sklearn.metrics import confusion_matrix
cm_ada = confusion_matrix(y_pred_ada,y_test)
print(cm_ada)

# CatBoost Kullanımı
from catboost import Pool, CatBoostClassifier
model_cat = CatBoostClassifier(iterations=100,
                               learning_rate=0.01,
                               depth=2,
                               loss_function='MultiClass')
model_cat.fit(x_train,y_train)

y_pred_cat = model_cat.predict(x_test)
y_pred_cat_proba = model_cat.predict_proba(x_test)
y_pred_cat_raw = model_cat.predict(x_test,
                                   prediction_type='RawFormulaVal')

from sklearn.metrics import confusion_matrix
cm_cat = confusion_matrix(y_pred_cat,y_test)
print(cm_cat)
print("###################### Conf. Matrisler. ##########################")

print('XGBoost Sonuçları:')
print(cm) #2830
print('AdaBoost Sonuçları:')
print(cm_ada)#2830
print('CatBoost Sonuçları:')
print(cm_cat)#2702


