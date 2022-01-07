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

# veri yükleme:
veriler = pd.read_csv("maaslar.csv")
print(veriler)

# dataframe dilimleme (slice):
x = veriler.iloc[:, 1:2]
y = veriler.iloc[:, 2:]

# numpy array'ine çevirme:
X = x.values
Y = y.values

# LINEAR REGRESSION:
# doğrusal model oluşturma:
from sklearn.linear_model import LinearRegression
ln_reg = LinearRegression()
ln_reg.fit(x,y)

# POLYNOMIAL REGRESSION:
# doğrusal olmayan (nonlinear model) model oluşturma:
# 2. dereceden polynomial:
from sklearn.preprocessing import PolynomialFeatures
poly_reg2 = PolynomialFeatures(degree=2) 
x_poly2 = poly_reg2.fit_transform(X)
ln_reg2 = LinearRegression()
ln_reg2.fit(x_poly2,y)

# 4. dereceden polynomial:
poly_reg4 = PolynomialFeatures(degree=4) 
x_poly4 = poly_reg4.fit_transform(X)
ln_reg4 = LinearRegression()
ln_reg4.fit(x_poly4,y)

# GÖRSELLEŞTİRME:
# 1. derece regression
plt.scatter(x, y)
plt.plot(x, ln_reg.predict(x))

# 2. derece regression
plt.scatter(X,Y)
plt.plot(X,ln_reg2.predict(poly_reg2.fit_transform(X)))
plt.show()

# 4. derece regression
plt.scatter(X,Y)
plt.plot(X,ln_reg4.predict(poly_reg4.fit_transform(X)))
plt.show()

# TAHMİNLER:
# linear regression daki modelin aynısı kullanıyoruz. 
# sadece tahmine vermeden önce polynomial dünyaya çeviriyoruz..

# 1. derece regression tahmini:
print(ln_reg.predict([[6.6]]))
print(ln_reg.predict([[11]]))

# 2. derece regression tahmini:
print(ln_reg2.predict(poly_reg2.fit_transform([[6.6]])))
print(ln_reg2.predict(poly_reg2.fit_transform([[11]])))

# 4. derece regression tahmini:
print(ln_reg4.predict(poly_reg4.fit_transform([[6.6]])))
print(ln_reg4.predict(poly_reg4.fit_transform([[11]])))

# SVR için verilerin scaler(ölçeklendirilmesi) işlemi:
from sklearn.preprocessing import StandardScaler
scX = StandardScaler()
x_olcekli = scX.fit_transform(X)
scY = StandardScaler()
y_olcekli = scY.fit_transform(Y)

# SVR MODELLENMESİ:
from sklearn.svm import SVR
svr_reg = SVR(kernel='rbf')
svr_reg.fit(x_olcekli, y_olcekli)

plt.scatter(x_olcekli,y_olcekli, color='red')
plt.plot(x_olcekli, svr_reg.predict(x_olcekli),color = 'blue') #y_olcekli ile  svr_reg.predict(x_olcekli) karşılaştırılmış olacak

print(svr_reg.predict([[6.6]]))
print(svr_reg.predict([[11]]))

