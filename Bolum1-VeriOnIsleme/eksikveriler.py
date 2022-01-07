# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.impute import SimpleImputer

eksikveriler = pd.read_csv("eksikveriler.csv")
print(eksikveriler)
imputer = SimpleImputer(missing_values=np.NAN,strategy="mean")


Yas = eksikveriler.iloc[:,1:4].values
print(Yas)

imputer = imputer.fit(Yas[:,1:4]) # simpleImputer daki stratejimiz mean olduğunu için bu kolondaki ortalama değerleri öğrenecek.
Yas[:,1:4] = imputer.transform(Yas[:,1:4])
print(Yas)