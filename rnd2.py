import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt 

veriler=pd.read_csv("Ads_CTR_Optimisation.csv")
print(veriler)

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

