import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt 

veriler=pd.read_csv("Ads_CTR_Optimisation.csv")
print(veriler)




#random selection (rastgele seçim)
"""
import random

N=1000
d=10
toplam=0
secilenler=[]
for n in range(0,N):
    ad=random.randrange(d)
    secilenler.append(ad)
    odul=veriler.values[n,ad]# verideki n,satir=1 ise odul 1
    toplam=toplam+odul

plt.hist(secilenler)
plt.show()
"""
import random
#ucb
N = 10000 # 10.000 tıklama
d = 10  # toplam 10 ilan var
#Ri(n)
oduller = [0] * d #ilk basta butun ilanların odulu 0
#Ni(n)

toplam = 0 # toplam odul
secilenler = []
birler=[0]*d
sifirlar=[0]*d
for n in range(1,N):
    ad = 0 #seçilen ilan
    max_th = 0
    for i in range(0,d):
        rasbeta=random.betavariate(birler[i]+1,sifirlar[i]+1)
        if rasbeta>max_th:
            max_th=rasbeta
            ad=i
            
       
            
         
    secilenler.append(ad)
    
    odul = veriler.values[n,ad] # verilerdeki n. satır = 1 ise odul 1
    if odul==1:
        birler[ad]=birler[ad]+1
    else:
        sifirlar[ad]=sifirlar[ad]+1    
    oduller[ad] = oduller[ad]+ odul
    toplam = toplam + odul
print('Toplam Odul:')   
print(toplam)

plt.hist(secilenler)
plt.show()
