#kütüphane
import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt 


#kodlar

#veri yükleme
veriler=pd.read_csv("C:/Users/Acer/Desktop/pythonmakineögrenme/veriler.csv")
#print(veriler)
#veri işleme
boy=veriler[["boy"]]
#print(boy)
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
    

l=[1,3,4]#liste


    