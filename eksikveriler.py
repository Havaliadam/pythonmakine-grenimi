# -*- coding: utf-8 -*-
"""
Created on Sat Oct 14 16:07:59 2023

@author: Acer
"""

import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt 


#kodlar

#veri yükleme
veriler=pd.read_csv("C:/Users/Acer/Desktop/pythonmakineögrenme/eksikveriler.csv")
print(veriler)
#veri işleme
boy=veriler[["boy"]]
print(boy)
boykilo=veriler[["boy","kilo"]]
print(boykilo)
x=10
class insan:
    boy=180
    def kosmak(self,b):
        return b+10
    
ali=insan()
print(ali.boy)
print(ali.kosmak(90))  
#eksik veriler


from sklearn.impute import SimpleImputer

imputer=SimpleImputer(missing_values=np.nan,strategy="mean")
yas=veriler.iloc[:1:4].values
print(yas)
imputer=imputer.fit(yas[:,1:4])
yas[:,1:4]=imputer.transform(yas[:,1:4])
print(yas)





