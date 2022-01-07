# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

veriler = pd.read_csv("veriler.csv")
print(veriler)

kilo = veriler["kilo"]
boykilo = veriler[["boy", "kilo"]]
print(boykilo)