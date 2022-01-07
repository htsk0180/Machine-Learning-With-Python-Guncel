# -*- coding: utf-8 -*-
"""
Created on Thu Jan  6 10:20:38 2022

@author: 104863
"""

import pandas as pd
url = "https://bilkav.com/satislar.csv"

dataset = pd.read_csv(url)
X = dataset.iloc[: , 0:1]
Y = dataset.iloc[: , 1]

bolme = 0.33

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(X,Y, test_size=bolme,random_state=0)
 
from sklearn.linear_model import LinearRegression
lr = LinearRegression()

lr.fit(x_train,y_train)
print(lr.predict(x_test))

##############################################################################
# Modeli Save etmek
import pickle
dosya = "model-kayit"
pickle.dump(lr, open(dosya, 'wb'))

# Save edilen modeli kullanmak.
yuklenen = pickle.load(open(dosya, 'rb'))
print(yuklenen.predict(x_test))


